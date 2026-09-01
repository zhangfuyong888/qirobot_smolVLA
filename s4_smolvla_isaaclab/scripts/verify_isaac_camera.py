#!/usr/bin/env python3
"""Render one real RGB frame for the Docker rollout verification profile."""

from __future__ import annotations

from isaaclab.app import AppLauncher


def main() -> int:
    simulation_app = AppLauncher(headless=True, enable_cameras=True).app
    print("[VERIFY][ISAAC] AppLauncher ready", flush=True)
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg

        sim_utils.create_new_stage()
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
        sim = sim_utils.SimulationContext(sim_cfg)
        sim_utils.update_stage()

        camera_cfg = TiledCameraCfg(
            height=128,
            width=128,
            prim_path="/World/Camera",
            update_period=0,
            data_types=["rgb"],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 4.0),
                rot=(0.0, 0.0, 1.0, 0.0),
                convention="ros",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
        )
        camera = TiledCamera(camera_cfg)
        sim.reset()

        for _ in range(5):
            sim.step()

        for _ in range(5):
            sim.step()
            camera.update(0.01)

        rgb = camera.data.output["rgb"]
        assert rgb is not None, "camera RGB output is None"
        assert rgb.numel() > 0, "camera RGB output is empty"
        shape = tuple(int(value) for value in rgb.shape)
        assert len(shape) == 4, shape
        assert shape[0] >= 1 and shape[1] > 0 and shape[2] > 0, shape
        assert shape[3] in (3, 4), shape
        print("[OK] Isaac Sim headless renderer", flush=True)
        print(f"[OK] Isaac Sim camera RGB frame {shape}", flush=True)
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
