"""Preflight checks for Conda and direct system-Python Pink runtimes."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import site
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RUNTIME_VERSIONS = {
    "scipy": "1.15.2",
    "aiohttp": "3.14.3",
    "qpsolvers": "4.12.0",
    "daqp": "0.8.7",
    "quadprog": "0.1.13",
}


def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except BaseException as exc:
        raise RuntimeError(
            f"required Python module {name!r} failed to import: {type(exc).__name__}: {exc}"
        ) from exc


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required Python distribution is missing: {name}") from exc


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_runtime_imports(*, robot_profile: bool) -> tuple[Any, Any, Any]:
    numpy = _module("numpy")
    scipy = _module("scipy")
    aiohttp = _module("aiohttp")
    pinocchio = _module("pinocchio")
    qpsolvers = _module("qpsolvers")
    daqp = _module("daqp")
    quadprog = _module("quadprog")

    versions = {
        "numpy": str(numpy.__version__),
        "pinocchio": str(pinocchio.__version__),
        "scipy": str(scipy.__version__),
        "aiohttp": str(aiohttp.__version__),
        "qpsolvers": str(qpsolvers.__version__),
        "daqp": _distribution_version("daqp"),
        "quadprog": _distribution_version("quadprog"),
    }
    print(
        f"[HW-PINK][DOCTOR] python={sys.version.split()[0]} "
        f"executable={sys.executable} packages={versions}"
    )
    print(f"[HW-PINK][DOCTOR] pinocchio_path={pinocchio.__file__}")

    for package, expected in EXPECTED_RUNTIME_VERSIONS.items():
        actual = versions[package]
        if actual != expected:
            raise RuntimeError(
                f"{package} version mismatch: expected {expected}, got {actual}"
            )

    runtime = os.environ.get("S4_HW_TELEOP_RUNTIME_RESOLVED", "unknown")
    if robot_profile:
        if runtime != "system":
            raise RuntimeError(
                f"robot profile requires system runtime, resolved runtime is {runtime!r}"
            )
        if sys.version_info[:2] != (3, 10):
            raise RuntimeError(
                f"robot profile requires Python 3.10, got {sys.version.split()[0]}"
            )
        expected_pin_prefix = Path(
            os.environ.get("HW_TELEOP_EXPECT_PIN_PREFIX", "/opt/ros/humble")
        )
        if not _is_relative_to(Path(pinocchio.__file__), expected_pin_prefix):
            raise RuntimeError(
                "robot profile selected the wrong Pinocchio: "
                f"expected under {expected_pin_prefix}, got {pinocchio.__file__}"
            )
        local_packages = Path(
            os.environ.get(
                "S4_HW_TELEOP_SITE_PACKAGES",
                str(PROJECT_ROOT / ".local/hardware_python"),
            )
        )
        user_packages = Path(site.getusersitepackages())
        allow_user_site = os.environ.get("S4_HW_TELEOP_ALLOW_USER_SITE", "0") == "1"
        local_modules = (
            ("scipy", scipy),
            ("aiohttp", aiohttp),
            ("qpsolvers", qpsolvers),
            ("daqp", daqp),
            ("quadprog", quadprog),
        )
        for name, module in local_modules:
            module_path = Path(module.__file__)
            from_local = _is_relative_to(module_path, local_packages)
            from_allowed_user_site = allow_user_site and _is_relative_to(
                module_path, user_packages
            )
            if not from_local and not from_allowed_user_site:
                raise RuntimeError(
                    f"robot profile must load {name} from project-local packages "
                    f"{local_packages}"
                    + (
                        f" or explicitly allowed user site {user_packages}"
                        if allow_user_site
                        else ""
                    )
                    + f", got {module.__file__}"
                )
            if from_allowed_user_site:
                print(
                    f"[HW-PINK][DOCTOR][WARN] {name} is loaded from user site: {module_path}"
                )
    return numpy, pinocchio, qpsolvers


def _validate_ros_environment(*, robot_profile: bool) -> Any:
    rclpy = _module("rclpy")
    try:
        from qi.msg import HandCmd, HandsCmd, LowCmd, LowState, MotorCmd
    except ImportError as exc:
        raise RuntimeError(f"ROS2/qi Python import failed: {exc}") from exc

    expected_fields = {
        HandCmd: {
            "positions": "uint16[6]",
            "durations": "uint16[6]",
            "mode": "uint8",
            "hand_id": "uint8",
        },
        HandsCmd: {
            "hands": "qi/HandCmd[2]",
            "mode": "uint8",
            "mode_ctrl": "uint8",
            "timestamp": "uint64",
        },
        LowCmd: {
            "motors": "sequence<qi/MotorCmd>",
            "mode": "uint8",
            "mode_ak": "uint8",
            "mode_ctrl": "uint8",
            "timestamp": "uint64",
        },
    }
    for message, expected in expected_fields.items():
        actual = message.get_fields_and_field_types()
        if actual != expected:
            raise RuntimeError(
                f"qi/{message.__name__} schema mismatch: expected={expected}, actual={actual}"
            )
    if not LowState.get_fields_and_field_types().get("motors", "").startswith("sequence<"):
        raise RuntimeError("qi/LowState schema is missing the motor sequence")
    if "reserve" not in MotorCmd.get_fields_and_field_types():
        raise RuntimeError("qi/MotorCmd schema is missing reserve")

    rmw = os.environ.get("RMW_IMPLEMENTATION", "")
    ros_domain = os.environ.get("ROS_DOMAIN_ID", "0")
    print(
        f"[HW-PINK][DOCTOR] ros_distro={os.environ.get('ROS_DISTRO', '')} "
        f"rmw={rmw} domain={ros_domain} qi_schema=ok"
    )
    if robot_profile:
        expected_domain = os.environ.get("HW_TELEOP_EXPECT_ROS_DOMAIN_ID", "16")
        if ros_domain != expected_domain:
            raise RuntimeError(
                f"robot ROS_DOMAIN_ID mismatch: expected {expected_domain}, got {ros_domain}"
            )
        if rmw != "rmw_cyclonedds_cpp":
            raise RuntimeError(
                "robot profile requires rmw_cyclonedds_cpp, got " + (rmw or "unset")
            )
        expected_interface = os.environ.get("HW_TELEOP_EXPECT_DDS_INTERFACE", "lo")
        cyclone_uri = os.environ.get("CYCLONEDDS_URI", "")
        interface_marker = f'name="{expected_interface}"'
        if interface_marker not in cyclone_uri:
            raise RuntimeError(
                f"CycloneDDS is not bound to expected interface {expected_interface!r}: "
                f"CYCLONEDDS_URI={cyclone_uri!r}"
            )
    print("[HW-PINK][DOCTOR] ROS2 and complete vendored qi messages import successfully")
    return rclpy


def _probe_solver(qpsolvers: Any, numpy: Any, name: str) -> tuple[bool, str]:
    if name not in qpsolvers.available_solvers:
        return False, "not installed"
    try:
        result = qpsolvers.solve_qp(
            numpy.eye(2, dtype=numpy.float64),
            numpy.array([-1.0, -1.0], dtype=numpy.float64),
            G=numpy.eye(2, dtype=numpy.float64),
            h=numpy.ones(2, dtype=numpy.float64),
            solver=name,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if result is None or not numpy.isfinite(result).all():
        return False, f"invalid result: {result}"
    return True, f"x={numpy.round(result, 6).tolist()}"


def _check_live_ros_graph(
    rclpy: Any,
    *,
    lowstate_topic: str,
    lowcmd_topic: str,
    discovery_timeout_s: float = 2.0,
) -> None:
    initialized_here = not rclpy.ok()
    if initialized_here:
        rclpy.init(args=None)
    node = rclpy.create_node("hardware_pink_doctor_read_only")
    try:
        deadline = time.monotonic() + max(float(discovery_timeout_s), 0.0)
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        topics = dict(node.get_topic_names_and_types())
        normalized_state = lowstate_topic if lowstate_topic.startswith("/") else f"/{lowstate_topic}"
        normalized_cmd = lowcmd_topic if lowcmd_topic.startswith("/") else f"/{lowcmd_topic}"
        if normalized_state not in topics:
            raise RuntimeError(
                f"live robot state topic is missing: {normalized_state}; visible={sorted(topics)}"
            )
        state_types = topics[normalized_state]
        if "qi/msg/LowState" not in state_types:
            raise RuntimeError(
                f"{normalized_state} has wrong type: expected qi/msg/LowState, got {state_types}"
            )
        publishers = node.get_publishers_info_by_topic(normalized_cmd)
        labels = sorted(
            {f"{info.node_namespace}/{info.node_name}".replace("//", "/") for info in publishers}
        )
        print(
            f"[HW-PINK][DOCTOR] live_state={normalized_state} type=qi/msg/LowState "
            f"lowcmd_publishers={labels}"
        )
        if len(labels) == 1:
            print(
                "[HW-PINK][DOCTOR] one lowcmd graph publisher is present; this is the "
                "expected standing-policy topology, but packet contents are checked next."
            )
        elif len(labels) > 1:
            print(
                "[HW-PINK][DOCTOR][WARN] multiple lowcmd publishers are present. Identify "
                "and stop old teleop, MoveIt or replay controllers before command output."
            )
    finally:
        node.destroy_node()
        if initialized_here and rclpy.ok():
            rclpy.shutdown()


def run_checks(
    *,
    require_daqp: bool = False,
    robot_profile: bool = False,
    require_live_state: bool = False,
    hardware_config: Path | None = None,
) -> int:
    forbidden = sorted(
        {name.split(".")[0] for name in sys.modules}
        & {"torch", "isaaclab", "isaacsim", "omni"}
    )
    if forbidden:
        raise RuntimeError(f"heavy runtime modules were imported: {forbidden}")

    numpy, _pinocchio, qpsolvers = validate_runtime_imports(robot_profile=robot_profile)
    rclpy = _validate_ros_environment(robot_profile=robot_profile)

    quadprog_ok, quadprog_detail = _probe_solver(qpsolvers, numpy, "quadprog")
    daqp_ok, daqp_detail = _probe_solver(qpsolvers, numpy, "daqp")
    print(f"[HW-PINK][DOCTOR] quadprog ok={quadprog_ok} {quadprog_detail}")
    print(f"[HW-PINK][DOCTOR] daqp ok={daqp_ok} {daqp_detail}")
    if not quadprog_ok:
        raise RuntimeError(f"quadprog preflight failed: {quadprog_detail}")
    if require_daqp and not daqp_ok:
        raise RuntimeError(f"DAQP preflight failed: {daqp_detail}")

    from hardware_teleop.config_loader import load_hardware_teleop_config
    from hardware_teleop.ik import create_pure_hardware_ik_backend
    from hardware_teleop.joint_mapping import bimanual_to_arm_q14
    from s4_robot.control_mapping import bimanual_default_action

    config_path = hardware_config or (
        PROJECT_ROOT / "hardware_teleop/config/quest_hardware.yaml"
    )
    config = load_hardware_teleop_config(config_path)
    topics = config.hardware
    if topics.lowstate_topic.lstrip("/").startswith("rt/"):
        raise RuntimeError(
            "hardware.lowstate_topic must be a ROS name such as 'lowstate', not native DDS 'rt/lowstate'"
        )
    print(
        f"[HW-PINK][DOCTOR] ros_topics state={topics.lowstate_topic!r} "
        f"command={topics.lowcmd_topic!r} hands={topics.hands_cmd_topic!r}; "
        "CycloneDDS native names receive the rt/ prefix automatically"
    )

    backend = create_pure_hardware_ik_backend(config)
    q14 = bimanual_to_arm_q14(bimanual_default_action())
    left, right = backend.forward(q14)
    print(
        f"[HW-PINK][DOCTOR] fk_left={numpy.round(left.position, 6).tolist()} "
        f"fk_right={numpy.round(right.position, 6).tolist()}"
    )
    backend.set_posture_reference(q14)
    samples_ms = []
    for index in range(320):
        start = time.perf_counter()
        backend.compute(q14, 1.0 / 30.0, left, right)
        if index >= 20:
            samples_ms.append((time.perf_counter() - start) * 1000.0)
    p50, p99 = numpy.percentile(numpy.asarray(samples_ms), [50.0, 99.0])
    print(f"[HW-PINK][DOCTOR] qp_compute_ms p50={p50:.4f} p99={p99:.4f}")
    if p99 >= 5.0:
        raise RuntimeError(f"Pink QP p99 is too slow for hardware target: {p99:.4f} ms")
    if require_live_state:
        _check_live_ros_graph(
            rclpy,
            lowstate_topic=topics.lowstate_topic,
            lowcmd_topic=topics.lowcmd_topic,
        )
        from hardware_teleop.ros import HardwareRobotBridge

        bridge = HardwareRobotBridge(
            config.hardware,
            config.hands,
            gravity_cfg=config.gravity,
            startup_cfg=config.startup,
            project_root=config.project_root,
            check_lowcmd_publishers=False,
            command_output_enabled=False,
        )
        try:
            bridge.wait_for_initial_state(config.hardware.initial_state_timeout_s)
            if config.startup.require_policy_lowcmd:
                bridge.wait_for_policy_lowcmd(
                    config.startup.policy_initial_timeout_s,
                    config.startup.policy_min_valid_frames,
                    config.startup.max_policy_age_s,
                    config.startup.policy_stable_duration_s,
                )
            print(f"[HW-PINK][DOCTOR] live_packets={bridge.diagnostics()}")
        finally:
            bridge.close()
        if robot_profile and config.startup.require_sdk_mode5_merge:
            from hardware_teleop.safety import find_verified_mode5_sdk_process

            sdk_pid, sdk_executable = find_verified_mode5_sdk_process(
                approved_sha256=config.startup.approved_sdk_sha256,
            )
            print(
                f"[HW-PINK][DOCTOR] sdk_mode5_merge=verified pid={sdk_pid} "
                f"executable={sdk_executable}"
            )
    print("[HW-PINK][DOCTOR] PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-daqp", action="store_true")
    parser.add_argument(
        "--robot-profile",
        action="store_true",
        help="Require the inspected S4 robot system-Python, ROS and project-local package layout.",
    )
    parser.add_argument(
        "--require-live-state",
        action="store_true",
        help="Read the ROS graph and require a qi/msg/LowState topic; creates no publisher.",
    )
    parser.add_argument(
        "--hardware-config",
        type=Path,
        default=PROJECT_ROOT / "hardware_teleop/config/quest_hardware.yaml",
    )
    args = parser.parse_args(argv)
    return run_checks(
        require_daqp=args.require_daqp,
        robot_profile=args.robot_profile,
        require_live_state=args.require_live_state,
        hardware_config=args.hardware_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
