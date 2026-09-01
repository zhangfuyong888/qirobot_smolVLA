from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DOCKER_RUN = WORKSPACE_ROOT / "docker/run.sh"
DOCKER_VERIFY = WORKSPACE_ROOT / "docker/verify_runtime.sh"


def test_docker_gpu_wrapper_builds_device_request_and_train_profile(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "docker-args.txt"
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$DOCKER_ARGS_CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "DOCKER_ARGS_CAPTURE": str(capture),
        }
    )

    result = subprocess.run(
        ["bash", str(DOCKER_RUN), "--gpus", "4,5", "verify-train"],
        cwd=WORKSPACE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    args = capture.read_text(encoding="utf-8").splitlines()
    gpu_flag = args.index("--gpus")
    assert args[gpu_flag + 1] == "device=4,5"
    assert "S4_DOCKER_SELECTED_GPU_COUNT=2" in args
    assert args[-3:] == ["s4-verify-runtime", "--profile", "train"]


def test_docker_gpu_wrapper_rejects_invalid_selection_without_running_docker(
    tmp_path: Path,
) -> None:
    fake_docker = tmp_path / "docker"
    fake_docker.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
    fake_docker.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:/usr/bin:/bin"
    result = subprocess.run(
        ["/bin/bash", str(DOCKER_RUN), "--gpus", "0,a", "verify"],
        cwd=WORKSPACE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Invalid GPU spec" in result.stderr


def test_docker_compose_wrapper_forwards_image_and_selected_gpu_count(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "docker-compose.txt"
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "{\n"
        "  printf 'S4_IMAGE_INTERNAL=%s\\n' \"${S4_IMAGE_INTERNAL:-}\"\n"
        "  printf 'S4_GPUS_INTERNAL=%s\\n' \"${S4_GPUS_INTERNAL:-}\"\n"
        "  printf 'S4_SELECTED_GPU_COUNT_INTERNAL=%s\\n' "
        "\"${S4_SELECTED_GPU_COUNT_INTERNAL:-}\"\n"
        "  printf 'ARG=%s\\n' \"$@\"\n"
        "} > \"$DOCKER_ARGS_CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "DOCKER_ARGS_CAPTURE": str(capture),
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(DOCKER_RUN),
            "--compose",
            "--image",
            "example/full-v4:test",
            "--gpus",
            "2,7",
            "verify-train",
        ],
        cwd=WORKSPACE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = capture.read_text(encoding="utf-8")
    assert "S4_IMAGE_INTERNAL=example/full-v4:test" in output
    assert "S4_GPUS_INTERNAL=2,7" in output
    assert "S4_SELECTED_GPU_COUNT_INTERNAL=2" in output
    assert "ARG=--gpus\nARG=device=2,7" in output
    assert "ARG=s4-verify-runtime\nARG=--profile\nARG=train" in output


def test_rollout_verifier_runs_camera_check_as_a_script() -> None:
    verifier = DOCKER_VERIFY.read_text(encoding="utf-8")
    assert 'isaac_camera_verify="$project_root/scripts/verify_isaac_camera.py"' in verifier
    assert '"$isaac_camera_verify" >"$kit_log" 2>&1' in verifier
    assert 'isaaclab_root/isaaclab.sh" -p -c' not in verifier
