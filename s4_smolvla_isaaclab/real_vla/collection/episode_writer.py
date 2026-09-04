"""Per-episode writer: async videos + low-dim streams. No RGB kept in RAM."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from real_vla.cameras.camera_device import CameraReader
from real_vla import SCHEMA_VERSION
from real_vla.collection.quality import QualityResult, evaluate_episode
from real_vla.collection.schema import CameraFrame, EpisodeMeta, PolicyState, PublishedCommand
from real_vla.collection.sync import alignment_report
from real_vla.config_loader import CollectionConfig


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _git_dirty(project_root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=str(project_root),
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        return result.returncode != 0 or bool(result.stdout.strip())
    except Exception:
        return True


def _source_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    roots = [
        project_root / "real_vla",
        project_root / "hardware_teleop",
        project_root / "teleoperation",
    ]
    files = [project_root / "run.sh"]
    for root in roots:
        files.extend(root.rglob("*.py"))
        files.extend(root.rglob("*.yaml"))
        files.extend(root.rglob("*.html"))
    for path in sorted({path for path in files if path.is_file()}):
        if "__pycache__" in path.parts or "ros_ws" in path.parts:
            continue
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def recover_orphaned_sessions(storage_root: Path) -> list[Path]:
    """Move pending episodes from previous, no-longer-live sessions aside."""
    recovered: list[Path] = []
    for path in sorted(storage_root.glob("session_*/pending/episode_*.tmp")):
        recovered_dir = path.parents[1] / "recovered"
        recovered_dir.mkdir(parents=True, exist_ok=True)
        meta_path = path / "meta.json"
        meta_payload: dict[str, Any] | None = None
        if meta_path.is_file():
            try:
                meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta_payload = None
        finalized = meta_payload is not None and (
            (path / "trajectory.h5").is_file()
            or (path / "trajectory.npz").is_file()
        )
        suffix = ".review" if finalized else ".incomplete"
        dest = recovered_dir / path.name.replace(".tmp", suffix)
        if dest.exists():
            dest = dest.with_name(dest.name + f".{int(time.time())}")
        shutil.move(str(path), str(dest))
        meta = dest / "meta.json"
        if meta_payload is not None:
            payload = meta_payload
            payload["result"] = (
                "recovered_pending_review" if finalized else "recovered_incomplete"
            )
            payload["recovered_path"] = str(dest)
            meta.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        else:
            meta.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "result": "recovered_incomplete",
                        "path": str(dest),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        manifest = dest.parents[1] / "manifest.jsonl"
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "episode": path.name,
                        "result": (
                            "recovered_pending_review"
                            if finalized
                            else "recovered_incomplete"
                        ),
                        "path": str(dest),
                    }
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        recovered.append(dest)
    return recovered


def _warm_video_codec(codec: str) -> None:
    """Load OpenCV encoder plugins before the 30 Hz arm loop starts."""
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="real_vla_codec_"))
    try:
        writer, _used = _open_video_writer(tmp / "warm.mkv", 16, 16, 30, codec)
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
        writer.release()
    except Exception:
        pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _open_video_writer(path: Path, width: int, height: int, fps: int, codec: str):
    import cv2

    candidates = []
    if codec.lower() in {"h264", "avc1", "x264"}:
        candidates.extend(["avc1", "H264", "X264", "mp4v", "MJPG"])
    else:
        candidates.extend(["MJPG", "mp4v"])
    for fourcc_name in candidates:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            float(fps),
            (int(width), int(height)),
        )
        if writer.isOpened():
            return writer, fourcc_name
        writer.release()
    raise RuntimeError(f"could not open video writer for {path}")


def _video_openable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        ok = bool(capture.isOpened())
        capture.release()
        return ok
    except Exception:
        return False


def _decoded_video_frames(path: Path) -> int:
    """Decode the complete stream so timestamps cannot outnumber real frames."""
    if not path.is_file() or path.stat().st_size <= 0:
        return 0
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        count = 0
        while capture.isOpened():
            ok, _frame = capture.read()
            if not ok:
                break
            count += 1
        capture.release()
        return count
    except Exception:
        return 0


def _write_trajectory(path: Path, payload: dict[str, Any]) -> str:
    try:
        import h5py
    except ImportError:
        np.savez_compressed(path.with_suffix(".npz"), **payload)
        return "npz"
    with h5py.File(path, "w") as handle:
        for key, value in payload.items():
            handle.create_dataset(key, data=value)
    return "h5"


@dataclass
class _VideoJob:
    frame: CameraFrame | None
    stop: bool = False


class _VideoWriterThread(threading.Thread):
    def __init__(
        self,
        *,
        name: str,
        path: Path,
        width: int,
        height: int,
        fps: int,
        codec: str,
        queue_size: int,
        timestamps_path: Path,
        seq_path: Path,
        drop_counter: list[int],
    ) -> None:
        super().__init__(name=f"video-writer-{name}", daemon=True)
        self.camera_name = name
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.queue: queue.Queue[_VideoJob] = queue.Queue(maxsize=max(int(queue_size), 1))
        self.timestamps_path = timestamps_path
        self.seq_path = seq_path
        self.drop_counter = drop_counter
        self.used_codec = ""
        self.error: str | None = None
        self.ready = threading.Event()
        self.stop_requested = threading.Event()

    def submit(self, frame: CameraFrame) -> bool:
        try:
            self.queue.put_nowait(_VideoJob(frame=frame))
            return True
        except queue.Full:
            self.drop_counter[0] += 1
            return False

    def request_stop(self) -> None:
        self.stop_requested.set()

    def run(self) -> None:
        writer = None
        try:
            writer, self.used_codec = _open_video_writer(
                self.path, self.width, self.height, self.fps, self.codec
            )
            self.ready.set()
            with self.timestamps_path.open("ab") as ts_file, self.seq_path.open("ab") as seq_file:
                while not self.stop_requested.is_set() or not self.queue.empty():
                    try:
                        job = self.queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if job.stop:
                        self.stop_requested.set()
                        continue
                    frame = job.frame
                    if frame is None:
                        continue
                    image = frame.image_bgr
                    if image.shape[1] != self.width or image.shape[0] != self.height:
                        import cv2

                        image = cv2.resize(image, (self.width, self.height))
                    writer.write(image)
                    ts_file.write(np.int64(frame.timestamp_ns).tobytes())
                    seq_file.write(np.int32(frame.capture_seq).tobytes())
                ts_file.flush()
                seq_file.flush()
                os.fsync(ts_file.fileno())
                os.fsync(seq_file.fileno())
        except Exception as exc:
            self.error = str(exc)
            self.ready.set()
        finally:
            if not self.ready.is_set():
                self.ready.set()
            if writer is not None:
                writer.release()


class EpisodeWriter:
    def __init__(self, config: CollectionConfig, session_dir: Path, project_root: Path) -> None:
        self.config = config
        self.session_dir = session_dir
        self.project_root = project_root
        self.episodes_dir = session_dir / "episodes"
        self.pending_dir = session_dir / "pending"
        self.recovered_dir = session_dir / "recovered"
        self.manifest_path = session_dir / "manifest.jsonl"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.recovered_dir.mkdir(parents=True, exist_ok=True)
        self.active_dir: Path | None = None
        self.episode_id = 0
        self.meta = EpisodeMeta()
        self._recording = False
        self._t_end_ns = 0
        self._state_path: Path | None = None
        self._action_path: Path | None = None
        self._video_threads: dict[str, _VideoWriterThread] = {}
        self._drop_counters: dict[str, list[int]] = {}
        self._camera_readers: dict[str, CameraReader] = {}
        self._lowdim_queue: queue.Queue[tuple[str, np.ndarray] | None] = queue.Queue(maxsize=4096)
        self._lowdim_stop = threading.Event()
        self._lowdim_thread: threading.Thread | None = None
        self._lowdim_drops = 0
        self._finalize_thread: threading.Thread | None = None
        self._finalized = threading.Event()
        self._finalize_error: str | None = None
        self._forced_invalid_notes: list[str] = []
        self.quality: QualityResult | None = None
        self._git_commit = _git_commit(project_root)
        self._git_dirty = _git_dirty(project_root)
        self._source_hash = _source_sha256(project_root)
        self._prepared = False
        self.prepare_error = ""
        _warm_video_codec(self.config.storage.video_codec)

    @property
    def is_prepared(self) -> bool:
        return bool(self._prepared) and not self._recording

    def recover_incomplete(self) -> list[Path]:
        moved: list[Path] = []
        live = self.active_dir.resolve() if self.active_dir is not None else None
        for path in sorted(self.pending_dir.glob("episode_*.tmp")):
            if live is not None and path.resolve() == live:
                continue
            dest = self.recovered_dir / path.name.replace(".tmp", "")
            dest = Path(str(dest) + ".incomplete")
            if dest.exists():
                dest = dest.with_name(dest.name + f".{int(time.time())}")
            shutil.move(str(path), str(dest))
            meta = dest / "meta.json"
            if not meta.is_file():
                payload = {
                    "schema_version": self.config.schema_version,
                    "result": "recovered_incomplete",
                    "path": str(dest),
                }
                meta.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            self._append_manifest(
                {
                    "episode": path.name,
                    "result": "recovered_incomplete",
                    "path": str(dest),
                }
            )
            moved.append(dest)
        return moved

    def next_episode_id(self) -> int:
        existing = [
            int(path.name.split("_")[1])
            for path in self.episodes_dir.glob("episode_*")
            if path.name.split("_")[-1].isdigit() or path.name.startswith("episode_")
        ]
        numbers = []
        for path in list(self.episodes_dir.glob("episode_*")) + list(self.pending_dir.glob("episode_*")):
            parts = path.name.replace(".tmp", "").split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                numbers.append(int(parts[1]))
        return (max(numbers) + 1) if numbers else 1

    def disk_gb(self) -> float:
        usage = shutil.disk_usage(self.session_dir)
        return usage.free / (1024 ** 3)

    def prepare_episode(
        self,
        episode_id: int,
        cameras: dict[str, CameraReader],
        *,
        wait_s: float = 15.0,
    ) -> Path:
        """Open video files and encoder threads. Must not run on the 30 Hz arm loop."""
        if self._recording:
            raise RuntimeError("cannot prepare while recording")
        self._prepared = False
        self.prepare_error = ""
        self.episode_id = int(episode_id)
        self.active_dir = self.pending_dir / f"episode_{self.episode_id:06d}.tmp"
        if self.active_dir.exists():
            shutil.rmtree(self.active_dir)
        self.active_dir.mkdir(parents=True)
        snapshot_dir = self.active_dir / "config_snapshot"
        snapshot_dir.mkdir()
        snapshot_sources = list((self.config.source_path.parent).glob("*.yaml"))
        snapshot_sources.append(self.config.hardware_teleop_config)
        for source in snapshot_sources:
            if source.is_file():
                name = source.name
                if source == self.config.hardware_teleop_config:
                    name = "hardware_teleop_" + name
                shutil.copy2(source, snapshot_dir / name)
        self._state_path = self.active_dir / "robot_state.bin"
        self._action_path = self.active_dir / "action.bin"
        self._lowdim_drops = 0
        self._lowdim_queue = queue.Queue(maxsize=4096)
        self._lowdim_stop.clear()
        self._lowdim_thread = threading.Thread(target=self._lowdim_loop, name="lowdim-writer", daemon=True)
        self._lowdim_thread.start()
        self._camera_readers = dict(cameras)
        self._drop_counters = {name: [0] for name in cameras}
        self._video_threads = {}
        self._t_end_ns = 0
        wrist_name = self.config.active_wrist_name
        for name, reader in cameras.items():
            filename = "head.mkv" if name == "head" else f"{wrist_name}.mkv"
            thread = _VideoWriterThread(
                name=name,
                path=self.active_dir / filename,
                width=reader.config.width,
                height=reader.config.height,
                fps=reader.config.fps,
                codec=self.config.storage.video_codec,
                queue_size=self.config.storage.writer_queue_frames,
                timestamps_path=self.active_dir / f"{name}_timestamp_ns.bin",
                seq_path=self.active_dir / f"{name}_capture_seq.bin",
                drop_counter=self._drop_counters[name],
            )
            self._video_threads[name] = thread
            thread.start()
        self.meta = EpisodeMeta(
            schema_version=self.config.schema_version,
            episode_id=self.episode_id,
            task=self.config.task.text,
            active_arm=self.config.active_arm,
            cameras=list(cameras.keys()),
            control_hz=self.config.robot.control_hz,
            git_commit=self._git_commit,
            git_dirty=self._git_dirty,
            source_sha256=self._source_hash,
            camera_specs={
                name: {
                    "serial": reader.config.serial,
                    "model": reader.config.model,
                    "width": reader.config.width,
                    "height": reader.config.height,
                    "fps": reader.config.fps,
                }
                for name, reader in cameras.items()
            },
            t_start_ns=0,
        )
        self._finalized.clear()
        self._finalize_error = None
        self._forced_invalid_notes = []
        self.quality = None
        deadline = time.monotonic() + max(float(wait_s), 0.1)
        for thread in self._video_threads.values():
            remaining = max(0.05, deadline - time.monotonic())
            if not thread.ready.wait(timeout=remaining):
                self.prepare_error = f"video writer {thread.camera_name} did not open in time"
                self._stop_prepared_writers()
                return self.active_dir
            if thread.error:
                self.prepare_error = f"{thread.camera_name}: {thread.error}"
                self._stop_prepared_writers()
                return self.active_dir
        self._prepared = True
        return self.active_dir

    def _stop_prepared_writers(self) -> None:
        self._lowdim_stop.set()
        for thread in self._video_threads.values():
            thread.request_stop()
        if self._lowdim_thread is not None:
            self._lowdim_thread.join(timeout=5.0)
        for thread in self._video_threads.values():
            thread.join(timeout=5.0)

    def cancel_prepared(self) -> None:
        """Close and remove an empty pre-opened episode during clean shutdown."""
        if self._recording or self._finalize_thread is not None:
            return
        self._prepared = False
        self._stop_prepared_writers()
        if self.active_dir is not None:
            shutil.rmtree(self.active_dir, ignore_errors=True)
        self.active_dir = None

    def begin_recording(self) -> None:
        """Arm capture. Cheap enough for the 30 Hz control thread."""
        if not self._prepared:
            raise RuntimeError(self.prepare_error or "recorder not prepared")
        if self._recording:
            raise RuntimeError("episode already recording")
        self._recording = True
        self._t_end_ns = 0
        self.meta.t_start_ns = time.monotonic_ns()
        for reader in self._camera_readers.values():
            reader.set_writer(self.accept_camera_frame)

    def start_episode(self, episode_id: int, cameras: dict[str, CameraReader]) -> Path:
        path = self.prepare_episode(episode_id, cameras)
        if not self._prepared:
            raise RuntimeError(self.prepare_error or f"could not prepare {path}")
        self.begin_recording()
        return path

    def accept_camera_frame(self, frame: CameraFrame) -> None:
        if not self._recording or (self._t_end_ns and frame.timestamp_ns > self._t_end_ns):
            return
        thread = self._video_threads.get(frame.name)
        if thread is None:
            return
        if not thread.submit(frame):
            reader = self._camera_readers.get(frame.name)
            if reader is not None:
                reader.buffer.writer_drops += 1

    def _submit_lowdim(self, kind: str, record: np.ndarray) -> None:
        try:
            self._lowdim_queue.put_nowait((kind, record.astype("<f8", copy=False)))
        except queue.Full:
            self._lowdim_drops += 1

    def record_state(self, state: PolicyState) -> None:
        if not self._recording or (self._t_end_ns and state.timestamp_ns > self._t_end_ns):
            return
        record = np.concatenate(
            [
                [float(state.timestamp_ns)],
                state.arm_q.astype(np.float64).reshape(7),
                [
                    float(state.gripper_state),
                    float(state.state_age_s),
                    float(state.valid),
                    float(state.phase),
                ],
            ]
        )
        self._submit_lowdim("state", record)

    def record_action(self, command: PublishedCommand) -> None:
        if not self._recording or (self._t_end_ns and command.timestamp_ns > self._t_end_ns):
            return
        record = np.concatenate(
            [
                [float(command.timestamp_ns)],
                command.arm_target_q.astype(np.float64).reshape(7),
                [float(command.gripper_target)],
                command.hand_command_6d.astype(np.float64).reshape(6),
                [
                    float(command.quest_trigger),
                    float(command.published),
                    float(command.limited),
                    float(command.motion_allowed),
                    float(command.fault_active),
                    float(command.input_valid),
                ],
            ]
        )
        self._submit_lowdim("action", record)

    def _lowdim_loop(self) -> None:
        while not self._lowdim_stop.is_set() or not self._lowdim_queue.empty():
            try:
                item = self._lowdim_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                self._lowdim_stop.set()
                continue
            kind, record = item
            path = self._state_path if kind == "state" else self._action_path
            if path is None:
                continue
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("ab") as handle:
                    handle.write(np.asarray(record, dtype="<f8").tobytes())
            except Exception as exc:
                self._lowdim_drops += 1
                print(f"[REAL-VLA] lowdim write failed ({kind}): {exc}", flush=True)

    def stop_accepting(self, t_end_ns: int) -> None:
        self._t_end_ns = int(t_end_ns)
        self._recording = False
        self._prepared = False
        for reader in self._camera_readers.values():
            reader.set_writer(None)
        for thread in self._video_threads.values():
            thread.request_stop()
        self._lowdim_stop.set()

    def mark_invalid(self, reason: str) -> None:
        message = str(reason).strip()
        if message and message not in self._forced_invalid_notes:
            self._forced_invalid_notes.append(message)

    def finalize_async(self) -> None:
        if self._finalize_thread is not None:
            return
        self._finalize_thread = threading.Thread(target=self._finalize_worker, name="episode-finalize", daemon=True)
        self._finalize_thread.start()

    def wait_finalized(self, timeout_s: float | None = None) -> bool:
        return self._finalized.wait(timeout=timeout_s)

    def _load_bin(self, path: Path, width: int) -> np.ndarray:
        if not path.is_file() or path.stat().st_size == 0:
            return np.zeros((0, width), dtype=np.float64)
        raw = np.fromfile(path, dtype="<f8")
        if raw.size % width != 0:
            raise RuntimeError(
                f"partial low-dimensional record in {path.name}: "
                f"{raw.size} values is not divisible by {width}"
            )
        return raw.reshape((-1, width))

    def _load_i64(self, path: Path) -> np.ndarray:
        if not path.is_file() or path.stat().st_size == 0:
            return np.zeros((0,), dtype=np.int64)
        return np.fromfile(path, dtype="<i8")

    def _load_i32(self, path: Path) -> np.ndarray:
        if not path.is_file() or path.stat().st_size == 0:
            return np.zeros((0,), dtype=np.int32)
        return np.fromfile(path, dtype="<i4")

    def _finalize_worker(self) -> None:
        try:
            if self._lowdim_thread is not None:
                self._lowdim_thread.join(timeout=5.0)
                if self._lowdim_thread.is_alive():
                    raise RuntimeError("low-dimensional writer did not stop")
            for thread in self._video_threads.values():
                thread.join(timeout=30.0)
                if thread.is_alive():
                    raise RuntimeError(f"video writer {thread.camera_name} did not stop")
            assert self.active_dir is not None
            state = self._load_bin(self.active_dir / "robot_state.bin", 12)
            action = self._load_bin(self.active_dir / "action.bin", 21)
            camera_ts = {}
            camera_seq = {}
            video_ok = {}
            decoded_frames = {}
            writer_drops = {}
            for name, thread in self._video_threads.items():
                camera_ts[name] = self._load_i64(self.active_dir / f"{name}_timestamp_ns.bin")
                camera_seq[name] = self._load_i32(self.active_dir / f"{name}_capture_seq.bin")
                video_ok[name] = _video_openable(thread.path) and thread.error is None
                decoded_frames[name] = _decoded_video_frames(thread.path)
                writer_drops[name] = int(self._drop_counters.get(name, [0])[0])
            duration_s = max(self._t_end_ns - self.meta.t_start_ns, 0) / 1.0e9
            quality = evaluate_episode(
                quality=self.config.quality,
                duration_s=duration_s,
                arm_q=state[:, 1:8] if state.size else np.zeros((0, 7)),
                action_q=action[:, 1:8] if action.size else np.zeros((0, 7)),
                state_ts=state[:, 0] if state.size else np.zeros((0,)),
                action_ts=action[:, 0] if action.size else np.zeros((0,)),
                camera_ts=camera_ts,
                camera_seq=camera_seq,
                writer_drops=writer_drops,
                video_ok=video_ok,
                decoded_video_frames=decoded_frames,
                state_valid=state[:, 10] if state.size else np.zeros((0,)),
                action_published=action[:, 16] if action.size else np.zeros((0,)),
                fault_active=action[:, 19] if action.size else np.zeros((0,)),
                input_valid=action[:, 20] if action.size else np.zeros((0,)),
                lowdim_drops=self._lowdim_drops,
                t_start_ns=self.meta.t_start_ns,
                t_end_ns=self._t_end_ns,
                forced_invalid_notes=self._forced_invalid_notes,
            )
            head_ts = camera_ts.get("head", np.zeros((0,), dtype=np.int64))
            wrist_key = next((key for key in camera_ts if key.startswith("wrist")), "wrist")
            wrist_ts = camera_ts.get(wrist_key, np.zeros((0,), dtype=np.int64))
            alignment = alignment_report(
                t_start_ns=self.meta.t_start_ns,
                t_end_ns=self._t_end_ns,
                head_ts=head_ts,
                wrist_ts=wrist_ts,
                state_ts=state[:, 0] if state.size else np.zeros((0,), dtype=np.int64),
                action_ts=action[:, 0] if action.size else np.zeros((0,), dtype=np.int64),
            )
            payload = {
                "robot_state_timestamp_ns": state[:, 0].astype(np.int64) if state.size else np.zeros((0,), dtype=np.int64),
                "robot_state_arm_q": state[:, 1:8] if state.size else np.zeros((0, 7)),
                "robot_state_gripper": state[:, 8:9] if state.size else np.zeros((0, 1)),
                "robot_state_age_s": state[:, 9:10] if state.size else np.zeros((0, 1)),
                "robot_state_valid": state[:, 10:11] if state.size else np.zeros((0, 1)),
                "collection_phase": state[:, 11:12] if state.size else np.zeros((0, 1)),
                "action_timestamp_ns": action[:, 0].astype(np.int64) if action.size else np.zeros((0,), dtype=np.int64),
                "action_arm_target_q": action[:, 1:8] if action.size else np.zeros((0, 7)),
                "action_gripper_target": action[:, 8:9] if action.size else np.zeros((0, 1)),
                "debug_hand_command_6d": action[:, 9:15] if action.size else np.zeros((0, 6)),
                "debug_quest_trigger": action[:, 15:16] if action.size else np.zeros((0, 1)),
                "action_published": action[:, 16:17] if action.size else np.zeros((0, 1)),
                "action_limited": action[:, 17:18] if action.size else np.zeros((0, 1)),
                "action_motion_allowed": action[:, 18:19] if action.size else np.zeros((0, 1)),
                "action_fault_active": action[:, 19:20] if action.size else np.zeros((0, 1)),
                "action_input_valid": action[:, 20:21] if action.size else np.zeros((0, 1)),
            }
            for name, stamps in camera_ts.items():
                payload[f"camera_{name}_timestamp_ns"] = stamps.astype(np.int64)
                payload[f"camera_{name}_capture_seq"] = camera_seq[name].astype(np.int32)
            fmt = _write_trajectory(self.active_dir / "trajectory.h5", payload)
            self.meta.t_end_ns = self._t_end_ns
            self.meta.duration_s = duration_s
            self.meta.quality_valid = quality.valid
            self.meta.quality_warning = quality.warning
            self.meta.quality_notes = list(quality.notes)
            self.meta.writer_drops = writer_drops
            self.meta.camera_stats = {
                name: reader.buffer.stats(time.monotonic_ns())
                for name, reader in self._camera_readers.items()
            }
            self.meta.alignment = alignment
            self.meta.result = "pending"
            meta_payload = asdict(self.meta)
            meta_payload["trajectory_format"] = fmt
            meta_payload["video_codecs"] = {
                name: thread.used_codec for name, thread in self._video_threads.items()
            }
            meta_payload["decoded_video_frames"] = decoded_frames
            meta_payload["lowdim_drops"] = self._lowdim_drops
            meta_payload["counts"] = {
                "robot_states": int(state.shape[0]),
                "actions": int(action.shape[0]),
                **{f"{name}_frames": int(stamps.shape[0]) for name, stamps in camera_ts.items()},
            }
            (self.active_dir / "meta.json").write_text(
                json.dumps(meta_payload, indent=2) + "\n", encoding="utf-8"
            )
            self.quality = quality
            self._finalized.set()
        except Exception as exc:
            self._finalize_error = str(exc)
            self._finalized.set()

    def save(self) -> Path:
        if not self._finalized.is_set():
            raise RuntimeError("cannot save before finalize")
        if self._finalize_error:
            raise RuntimeError(f"finalize failed: {self._finalize_error}")
        if self.quality is None or not self.quality.valid:
            raise RuntimeError("invalid episode cannot be saved; discard it instead")
        assert self.active_dir is not None
        dest = self.episodes_dir / f"episode_{self.episode_id:06d}"
        if dest.exists():
            raise RuntimeError(f"episode already exists: {dest}")
        meta_path = self.active_dir / "meta.json"
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        payload["result"] = "saved"
        meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        fd = os.open(str(meta_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_dir(self.active_dir)
        os.rename(self.active_dir, dest)
        _fsync_dir(self.episodes_dir)
        self._append_manifest(
            {
                "episode": self.episode_id,
                "result": "saved",
                "quality": "pass" if payload.get("quality_valid") and not payload.get("quality_warning") else (
                    "warn" if payload.get("quality_valid") else "invalid"
                ),
                "duration_s": payload.get("duration_s"),
                "path": str(dest),
            }
        )
        self.active_dir = None
        self._finalize_thread = None
        return dest

    def discard(self) -> None:
        duration_s = self.meta.duration_s
        if self.active_dir is not None and self.active_dir.exists():
            shutil.rmtree(self.active_dir, ignore_errors=True)
        self._append_manifest(
            {
                "episode": self.episode_id,
                "result": "discarded",
                "duration_s": duration_s,
            }
        )
        self.active_dir = None
        self._finalize_thread = None
        self._recording = False

    def review_summary(self) -> str:
        if self.active_dir is None:
            return "no pending episode"
        meta_path = self.active_dir / "meta.json"
        if not meta_path.is_file():
            return f"EP {self.episode_id:03d} finalizing..."
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        counts = payload.get("counts", {})
        quality = self.quality.label if self.quality is not None else "QUALITY UNKNOWN"
        lines = [
            f"EP {self.episode_id:03d}",
            f"Duration: {payload.get('duration_s', 0.0):.1f}s",
            f"Head: {counts.get('head_frames', 0)} frames",
            f"Wrist: {counts.get(self.config.active_wrist_name + '_frames', 0)} frames",
            f"Robot states: {counts.get('robot_states', 0)}",
            f"Actions: {counts.get('actions', 0)}",
            quality,
        ]
        notes = payload.get("quality_notes") or []
        lines.extend(notes[:4])
        lines.append("X = SAVE")
        lines.append("Y hold = DISCARD")
        return "\n".join(lines)

    def _append_manifest(self, row: dict[str, Any]) -> None:
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
