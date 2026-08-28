"""Isaac Lab scene construction and reset utilities for S4 grasping debug."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg, SimulationContext, schemas
from isaaclab.sim.spawners.shapes import CuboidCfg, CylinderCfg
from isaaclab.utils.math import matrix_from_euler, quat_from_euler_xyz, quat_from_matrix

from .s4_robot_cfg import (
    ALL_DRIVE_JOINTS,
    LEFT_HAND_MIMIC_JOINTS,
    RIGHT_HAND_MIMIC_JOINTS,
    URDF_PATH,
    get_default_joint_positions,
)
from .visuals import FINGER_BLUE_GRAY, is_finger_visual_mesh_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_ASSET_ISAAC_DIR = Path(
    os.environ.get(
        "S4_SCENE_ASSET_ROOT",
        PROJECT_ROOT / "local_assets" / "isaac" / "5.1",
    )
) / "Isaac"
DEFAULT_SCENE_USD = SCENE_ASSET_ISAAC_DIR / "Environments" / "Simple_Warehouse" / "warehouse.usd"
DEFAULT_TABLE_USD = SCENE_ASSET_ISAAC_DIR / "Props" / "PackingTable" / "packing_table.usd"
PILL_BOTTLE_USDZ = PROJECT_ROOT / "assets" / "scenes" / "Pill_Bottle.usdz"

BLOCK_CYLINDER_RADIUS = 0.035
BLOCK_CYLINDER_HEIGHT = 0.12
BLOCK_MASS = 0.08
PILL_BOTTLE_SCALE_VALUE = 0.001103
PILL_BOTTLE_SCALE = (PILL_BOTTLE_SCALE_VALUE, PILL_BOTTLE_SCALE_VALUE, PILL_BOTTLE_SCALE_VALUE)
PILL_BOTTLE_SIZE_M = (0.132, 0.120, 0.074)
PILL_BOTTLE_ORIENTATION = (0.7071068, 0.7071068, 0.0, 0.0)
PILL_BOTTLE_ROOT_Z_OFFSET = 0.005
PLATE_RADIUS = 0.13
TASK_PLATFORM_HEIGHT = 0.05
TASK_PLATFORM_SIZE = (0.56, 0.72, TASK_PLATFORM_HEIGHT)
TABLE_YAW_90_QUAT = (0.7071068, 0.0, 0.0, 0.7071068)
TASK_OBJECT_KEYS = ("task_platform", "red", "blue", "plate")
TABLE_CLUTTER_RELATIVE_PRIMS = ("container_h20",)
TABLE_CLUTTER_NAME_TOKENS = (
    "container",
    "corrugatedbox",
    "box_",
)
TABLE_CLUTTER_EXACT_PREFIXES = (
    "SM_Crate_A",
)
# Real hand-eye calibration gives hand_base_link -> camera optical frame.
# IsaacSim merges the fixed hand_base links into the wrist yaw links, so these
# defaults are URDF wrist_yaw_link -> hand_base_link composed with the measured
# hand_base_link -> camera transforms.
LEFT_WRIST_CAMERA_LOCAL_POS = (-0.0445941356, -0.0209877889, -0.1614989107)
LEFT_WRIST_CAMERA_LOCAL_QUAT_WXYZ = (-0.1871460184, 0.6595136840, 0.6044971537, 0.4057108079)
RIGHT_WRIST_CAMERA_LOCAL_POS = (0.0445948230, -0.0207078601, -0.1638273481)
RIGHT_WRIST_CAMERA_LOCAL_QUAT_WXYZ = (-0.1353444104, 0.6807588438, -0.5885558066, -0.4145495744)
WRIST_CAMERA_LOCAL_RPY_DEG: tuple[float, float, float] | None = None
WRIST_CAMERA_OFFSET_CONVENTION = "ros"


@dataclass(frozen=True)
class TaskLayout:
    """World-frame task coordinates.

    The robot is fixed at world origin. Positive X is the table direction.
    Keep all task objects around ``table_center_y`` so the visual table and
    physics objects do not drift into the robot feet area independently.
    """

    table_center_x: float = 0.82
    table_center_y: float = -0.05
    block_x: float = 0.50
    block_y_offset: float = 0.20
    plate_x: float = 0.50

    def task_surface_z(self, table_top_z: float) -> float:
        return table_top_z + TASK_PLATFORM_HEIGHT

    def task_platform_pos(self, table_top_z: float) -> np.ndarray:
        return np.array(
            [self.block_x, self.table_center_y, table_top_z + TASK_PLATFORM_HEIGHT * 0.5],
            dtype=np.float32,
        )

    def red_block_pos(self, table_top_z: float) -> np.ndarray:
        surface_z = self.task_surface_z(table_top_z)
        # Legacy name: the red task slot is now a scaled pill-bottle asset.
        # The USDZ is Y-up and authored in centimeters, so its root is placed
        # near the platform surface after the Y->Z rotation.
        return np.array(
            [
                self.block_x,
                self.table_center_y + self.block_y_offset,
                surface_z + PILL_BOTTLE_ROOT_Z_OFFSET,
            ],
            dtype=np.float32,
        )

    def blue_block_pos(self, table_top_z: float) -> np.ndarray:
        surface_z = self.task_surface_z(table_top_z)
        return np.array(
            [self.block_x, self.table_center_y - self.block_y_offset, surface_z + BLOCK_CYLINDER_HEIGHT * 0.5],
            dtype=np.float32,
        )

    def plate_pos(self, table_top_z: float) -> np.ndarray:
        return np.array([self.plate_x, self.table_center_y, self.task_surface_z(table_top_z) + 0.015], dtype=np.float32)


@dataclass(frozen=True)
class SceneBuildCfg:
    table_top_z: float
    joint_stiffness: float
    joint_damping: float
    joint_effort_limit: float
    robot_base_z: float = 1.08
    scene_usd: Path = DEFAULT_SCENE_USD
    table_usd: Path | None = DEFAULT_TABLE_USD
    table_visual_z: float = 0.0
    table_scale: float = 1.0
    clean_table_clutter: bool = True
    layout: TaskLayout = TaskLayout()
    camera_eye: tuple[float, float, float] = (0.10, 0.0, 1.80)
    camera_target: tuple[float, float, float] = (0.68, 0.0, 1.02)
    camera_rpy_deg: tuple[float, float, float] | None = (0.0, -23.0, -90.0)
    camera_convention: str = "opengl"
    camera_width: int = 680
    camera_height: int = 480
    left_wrist_camera_pos: tuple[float, float, float] = LEFT_WRIST_CAMERA_LOCAL_POS
    left_wrist_camera_quat_wxyz: tuple[float, float, float, float] = LEFT_WRIST_CAMERA_LOCAL_QUAT_WXYZ
    left_wrist_camera_rpy_deg: tuple[float, float, float] | None = WRIST_CAMERA_LOCAL_RPY_DEG
    right_wrist_camera_pos: tuple[float, float, float] = RIGHT_WRIST_CAMERA_LOCAL_POS
    right_wrist_camera_quat_wxyz: tuple[float, float, float, float] = RIGHT_WRIST_CAMERA_LOCAL_QUAT_WXYZ
    right_wrist_camera_rpy_deg: tuple[float, float, float] | None = WRIST_CAMERA_LOCAL_RPY_DEG
    wrist_camera_convention: str = WRIST_CAMERA_OFFSET_CONVENTION
    # Collection and rollout keep the default True. Teleop sets False to skip
    # unused 680x480 RGB sensors and reduce GPU load while keeping the viewport.
    spawn_rgb_cameras: bool = True


def create_simulation_context(device: str, *, use_fabric: bool = True) -> SimulationContext:
    sim = SimulationContext(
        SimulationCfg(
            device=device,
            dt=1.0 / 120.0,
            use_fabric=bool(use_fabric),
            physx=PhysxCfg(
                enable_ccd=True,
                enable_stabilization=True,
                enable_external_forces_every_iteration=True,
                solve_articulation_contact_last=True,
                min_position_iteration_count=4,
                min_velocity_iteration_count=1,
                gpu_max_rigid_contact_count=2**23,
                gpu_max_rigid_patch_count=2**18,
            ),
            # Fixed, quality-oriented RTX settings shared by preview, recording,
            # and policy rollout. These affect rendering only, not physics or
            # the observations/actions contract.
            render=RenderCfg(
                antialiasing_mode="DLAA",
                enable_reflections=True,
                enable_global_illumination=True,
                enable_direct_lighting=True,
                enable_dl_denoiser=True,
                samples_per_pixel=4,
                enable_shadows=True,
                enable_ambient_occlusion=True,
                dome_light_upper_lower_strategy=4,
            ),
        )
    )
    sim.set_camera_view([0.18, -0.62, 1.42], [0.52, -0.12, 0.98])
    return sim


def build_robot(
    prim_path: str,
    joint_stiffness: float,
    joint_damping: float,
    joint_effort_limit: float,
    robot_base_z: float,
) -> Articulation:
    robot_cfg = ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(URDF_PATH.resolve()),
            fix_base=True,
            merge_fixed_joints=True,
            self_collision=False,
            articulation_props=schemas.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
            joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=joint_stiffness,
                    damping=joint_damping,
                ),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, robot_base_z),
            joint_pos={j: float(v) for j, v in zip(ALL_DRIVE_JOINTS, get_default_joint_positions(), strict=True)},
        ),
        actuators={
            "drive_joints": ImplicitActuatorCfg(
                joint_names_expr=list(ALL_DRIVE_JOINTS),
                stiffness=joint_stiffness,
                damping=joint_damping,
                effort_limit_sim=joint_effort_limit,
            ),
            # Mimic joints are not independent policy DOFs, but the imported
            # articulation exposes them as normal joints and control_mapping
            # writes deterministic targets derived from the six hand controls.
            "mapped_hand_mimic_joints": ImplicitActuatorCfg(
                joint_names_expr=list(LEFT_HAND_MIMIC_JOINTS + RIGHT_HAND_MIMIC_JOINTS),
                stiffness=joint_stiffness,
                damping=joint_damping,
                effort_limit_sim=joint_effort_limit,
            ),
        },
    )
    robot = Articulation(cfg=robot_cfg)
    apply_finger_visual_material(prim_path)
    return robot


def apply_finger_visual_material(robot_prim_path: str) -> None:
    """Color only the two hands' finger visual meshes blue-gray.

    Collision meshes, palms, wrists, rigid-body properties, and articulation
    behavior are deliberately left untouched.
    """
    try:
        import omni.usd
        from pxr import Usd
        from isaaclab.sim.utils import bind_visual_material
    except Exception as exc:
        print(f"[WARN] could not configure finger visual material: {exc}", flush=True)
        return

    stage = omni.usd.get_context().get_stage()
    robot_root = stage.GetPrimAtPath(robot_prim_path)
    if not robot_root.IsValid():
        print(f"[WARN] robot root not found for finger material: {robot_prim_path}", flush=True)
        return

    material_path = "/World/Looks/S4FingerBlueGray"
    material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=FINGER_BLUE_GRAY,
        roughness=0.68,
        metallic=0.04,
    )
    if not stage.GetPrimAtPath(material_path).IsValid():
        material.func(material_path, material)

    finger_meshes = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(robot_root)
        if prim.GetTypeName() == "Mesh" and is_finger_visual_mesh_path(str(prim.GetPath()))
    ]
    for mesh_path in finger_meshes:
        bind_visual_material(mesh_path, material_path, stage=stage, stronger_than_descendants=True)
    if not finger_meshes:
        print("[WARN] no imported finger visual meshes matched the material selector", flush=True)
        return
    print(
        f"[BOOT] finger visuals colored blue-gray: meshes={len(finger_meshes)} "
        f"rgb={FINGER_BLUE_GRAY}",
        flush=True,
    )


def spawn_background_and_table(cfg: SceneBuildCfg) -> None:
    if not cfg.scene_usd.is_file():
        raise FileNotFoundError(f"Scene USD not found: {cfg.scene_usd}")

    print(f"[BOOT] loading background scene: {cfg.scene_usd}", flush=True)
    scene_cfg = sim_utils.UsdFileCfg(usd_path=str(cfg.scene_usd))
    scene_cfg.func("/World/BackgroundScene", scene_cfg)
    print("[BOOT] background scene loaded.", flush=True)
    configure_fixed_lighting()

    if cfg.table_usd is None:
        return
    if not cfg.table_usd.is_file():
        raise FileNotFoundError(f"Table USD not found: {cfg.table_usd}")

    print(f"[BOOT] loading table: {cfg.table_usd}", flush=True)
    table_cfg = sim_utils.UsdFileCfg(
        usd_path=str(cfg.table_usd),
        scale=(cfg.table_scale, cfg.table_scale, cfg.table_scale),
        rigid_props=schemas.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    table_cfg.func(
        "/World/TaskTableVisual",
        table_cfg,
        translation=(cfg.layout.table_center_x, cfg.layout.table_center_y, cfg.table_visual_z),
        orientation=TABLE_YAW_90_QUAT,
    )
    print("[BOOT] table loaded.", flush=True)
    if cfg.clean_table_clutter:
        remove_table_clutter("/World/TaskTableVisual")


def configure_fixed_lighting() -> None:
    """Keep fixed warehouse fixtures and add a deterministic task-area rig.

    The large warm overhead source provides soft, directional shadows; a
    neutral dome and weaker front fill prevent the robot, can, and open drawer
    from becoming unnaturally black. No value in this rig is randomized.
    """
    try:
        import omni.usd
        from pxr import Sdf, Usd, UsdGeom, UsdLux
    except Exception as exc:
        print(f"[WARN] could not configure fixed lighting: {exc}", flush=True)
        return

    stage = omni.usd.get_context().get_stage()
    lighting_root = stage.GetPrimAtPath("/World/S4Lighting")
    if lighting_root.IsValid():
        stage.RemovePrim(Sdf.Path("/World/S4Lighting"))

    authored_light_count = 0
    near_light_count = 0
    far_light_count = 0
    near_light_scale = 0.18
    far_light_scale = 0.55
    task_center_xy = np.asarray([0.80, 0.0], dtype=np.float64)
    near_light_radius_m = 3.5
    background = stage.GetPrimAtPath("/World/BackgroundScene")
    if background.IsValid():
        authored_lights = [prim for prim in Usd.PrimRange(background) if prim.HasAPI(UsdLux.LightAPI)]
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        # Keep fixtures close to the task dim to preserve white robot/table
        # texture. Raise only distant fixtures so the warehouse background is
        # bright without increasing direct illumination on the task area.
        for prim in authored_lights:
            light = UsdLux.LightAPI(prim)
            intensity_attr = light.GetIntensityAttr()
            intensity = intensity_attr.Get()
            if intensity is not None:
                if prim.GetTypeName() == "DistantLight":
                    scale = near_light_scale
                    near_light_count += 1
                else:
                    world_pos = xform_cache.GetLocalToWorldTransform(prim).ExtractTranslation()
                    distance_xy = float(
                        np.linalg.norm(np.asarray([world_pos[0], world_pos[1]]) - task_center_xy)
                    )
                    if distance_xy <= near_light_radius_m:
                        scale = near_light_scale
                        near_light_count += 1
                    else:
                        scale = far_light_scale
                        far_light_count += 1
                intensity_attr.Set(float(intensity) * scale)
        authored_light_count = len(authored_lights)

    dome_cfg = sim_utils.DomeLightCfg(
        color=(0.94, 0.97, 1.0),
        enable_color_temperature=True,
        color_temperature=5800.0,
        # The enclosed warehouse already has 18 strong ceiling fixtures. Keep
        # the dome low so it lifts deep shadows without flattening robot albedo
        # and roughness details.
        intensity=100.0,
        visible_in_primary_ray=False,
    )
    dome_cfg.func("/World/S4Lighting/AmbientDome", dome_cfg)

    fill_cfg = sim_utils.SphereLightCfg(
        color=(0.96, 0.98, 1.0),
        enable_color_temperature=True,
        color_temperature=6200.0,
        normalize=True,
        intensity=50.0,
        radius=0.45,
    )
    fill_cfg.func(
        "/World/S4Lighting/FrontFill",
        fill_cfg,
        translation=(0.0, -0.85, 1.9),
    )
    print(
        f"[BOOT] fixed lighting ready (warehouse lights: {authored_light_count}; "
        f"task-zone={near_light_count} at {near_light_scale:.0%}, "
        f"background={far_light_count} at {far_light_scale:.0%}; low ambient/fill).",
        flush=True,
    )


def remove_table_clutter(table_root_path: str) -> None:
    """Deactivate known top-level PackingTable clutter while keeping the table body visible."""
    try:
        import omni.usd
        from pxr import Usd
    except Exception as exc:
        print(f"[WARN] could not import omni.usd to remove table clutter: {exc}")
        return

    stage = omni.usd.get_context().get_stage()
    table_root = stage.GetPrimAtPath(table_root_path)
    if not table_root.IsValid():
        print(f"[WARN] table root not found for clutter cleanup: {table_root_path}")
        return

    removed: list[str] = []
    for rel_path in TABLE_CLUTTER_RELATIVE_PRIMS:
        path = f"{table_root_path}/{rel_path}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"[WARN] table clutter prim not found: {path}")
            continue
        try:
            prim.SetActive(False)
            removed.append(path)
        except Exception as exc:
            print(f"[WARN] could not deactivate table clutter prim {path}: {exc}")
    for prim in list(Usd.PrimRange(table_root)):
        if prim == table_root or not prim.IsValid() or not prim.IsActive():
            continue
        prim_name = prim.GetName()
        name = prim_name.lower()
        path = str(prim.GetPath())
        if any(token in name for token in TABLE_CLUTTER_NAME_TOKENS) or any(
            prim_name.startswith(prefix) for prefix in TABLE_CLUTTER_EXACT_PREFIXES
        ):
            try:
                prim.SetActive(False)
                removed.append(path)
            except Exception as exc:
                print(f"[WARN] could not deactivate table clutter prim {path}: {exc}")
    if removed:
        unique_removed = list(dict.fromkeys(removed))
        print(f"[INFO] Removed PackingTable clutter prims: {', '.join(unique_removed)}")


def configure_usdz_rigid_meshes(
    prim_path: str,
    mass_props: schemas.MassPropertiesCfg,
    rigid_props: schemas.RigidBodyPropertiesCfg,
    collision_props: schemas.CollisionPropertiesCfg,
    physics_material: sim_utils.RigidBodyMaterialCfg,
) -> None:
    """Apply simple rigid-body and convex mesh collision settings to an imported USDZ object."""
    try:
        from pxr import Usd

        from isaaclab.sim.utils import bind_physics_material
        from isaaclab.sim.utils.stage import get_current_stage
    except Exception as exc:
        print(f"[WARN] could not import USD helpers for {prim_path}: {exc}")
        return

    stage = get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        print(f"[WARN] USDZ rigid mesh root not found: {prim_path}")
        return

    try:
        schemas.define_rigid_body_properties(prim_path, rigid_props, stage=stage)
        schemas.define_mass_properties(prim_path, mass_props, stage=stage)
    except Exception as exc:
        print(f"[WARN] could not define rigid body properties on {prim_path}: {exc}")

    mesh_paths = [str(prim.GetPath()) for prim in Usd.PrimRange(root) if prim.GetTypeName() == "Mesh"]
    if not mesh_paths:
        print(f"[WARN] no mesh prims found under USDZ object: {prim_path}")
        return

    material_path = f"{prim_path}/physicsMaterial"
    try:
        physics_material.func(material_path, physics_material)
    except Exception as exc:
        print(f"[WARN] could not create physics material for {prim_path}: {exc}")
        material_path = ""

    for mesh_path in mesh_paths:
        try:
            schemas.define_collision_properties(mesh_path, collision_props, stage=stage)
            schemas.define_mesh_collision_properties(mesh_path, schemas.ConvexHullPropertiesCfg(), stage=stage)
            if material_path:
                bind_physics_material(mesh_path, material_path, stage=stage)
        except Exception as exc:
            print(f"[WARN] could not configure collision mesh {mesh_path}: {exc}")
    print(f"[BOOT] configured USDZ rigid mesh collisions: {prim_path} meshes={len(mesh_paths)}", flush=True)


def spawn_physics_task_objects(cfg: SceneBuildCfg) -> dict[str, RigidObject]:
    if not PILL_BOTTLE_USDZ.is_file():
        raise FileNotFoundError(f"Pill bottle USDZ not found: {PILL_BOTTLE_USDZ}")

    print("[BOOT] creating task object configs...", flush=True)
    contact_material = sim_utils.RigidBodyMaterialCfg(static_friction=2.0, dynamic_friction=1.6, restitution=0.0)
    collision_props = schemas.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0005)
    dynamic_rigid_props = schemas.RigidBodyPropertiesCfg(
        solver_position_iteration_count=24,
        solver_velocity_iteration_count=4,
        max_depenetration_velocity=0.25,
        linear_damping=0.25,
        angular_damping=0.35,
    )
    def make_platform_cfg(name: str, pos: np.ndarray, size: tuple[float, float, float]) -> RigidObjectCfg:
        return RigidObjectCfg(
            prim_path=f"/World/RecordTask/{name}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(x) for x in pos)),
            spawn=CuboidCfg(
                size=size,
                rigid_props=schemas.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=2,
                ),
                collision_props=collision_props,
                physics_material=contact_material,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.20, 0.18)),
            ),
        )

    task_platform_cfg = make_platform_cfg(
        "TaskPlatform",
        cfg.layout.task_platform_pos(cfg.table_top_z),
        TASK_PLATFORM_SIZE,
    )
    red_cfg = RigidObjectCfg(
        prim_path="/World/RecordTask/RedBlock",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(float(x) for x in cfg.layout.red_block_pos(cfg.table_top_z)),
            rot=PILL_BOTTLE_ORIENTATION,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PILL_BOTTLE_USDZ),
            scale=PILL_BOTTLE_SCALE,
            mass_props=schemas.MassPropertiesCfg(mass=BLOCK_MASS),
            rigid_props=dynamic_rigid_props,
            collision_props=collision_props,
        ),
    )
    blue_cfg = RigidObjectCfg(
        prim_path="/World/RecordTask/BlueBlock",
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(x) for x in cfg.layout.blue_block_pos(cfg.table_top_z))),
        spawn=CylinderCfg(
            radius=BLOCK_CYLINDER_RADIUS,
            height=BLOCK_CYLINDER_HEIGHT,
            mass_props=schemas.MassPropertiesCfg(mass=BLOCK_MASS),
            rigid_props=dynamic_rigid_props,
            collision_props=collision_props,
            physics_material=contact_material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.22, 1.0)),
        ),
    )
    plate_cfg = RigidObjectCfg(
        prim_path="/World/RecordTask/Plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(x) for x in cfg.layout.plate_pos(cfg.table_top_z))),
        spawn=CylinderCfg(
            radius=PLATE_RADIUS,
            height=0.025,
            rigid_props=schemas.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
            collision_props=collision_props,
            physics_material=contact_material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.92, 0.92, 0.86)),
        ),
    )

    print("[BOOT] instantiating task_platform...", flush=True)
    task_platform_obj = RigidObject(cfg=task_platform_cfg)
    print("[BOOT] instantiating red pill bottle...", flush=True)
    red_obj = RigidObject(cfg=red_cfg)
    configure_usdz_rigid_meshes(
        "/World/RecordTask/RedBlock",
        schemas.MassPropertiesCfg(mass=BLOCK_MASS),
        dynamic_rigid_props,
        collision_props,
        contact_material,
    )
    print("[BOOT] instantiating blue cylinder...", flush=True)
    blue_obj = RigidObject(cfg=blue_cfg)
    print("[BOOT] instantiating plate...", flush=True)
    plate_obj = RigidObject(cfg=plate_cfg)
    print("[BOOT] task objects ready.", flush=True)

    return {
        "task_platform": task_platform_obj,
        "red": red_obj,
        "blue": blue_obj,
        "plate": plate_obj,
    }


def make_rgb_camera(
    prim_path: str,
    cfg: SceneBuildCfg,
    *,
    offset: CameraCfg.OffsetCfg | None = None,
) -> Camera:
    return Camera(
        cfg=CameraCfg(
            prim_path=prim_path,
            update_period=0,
            height=int(cfg.camera_height),
            width=int(cfg.camera_width),
            data_types=["rgb"],
            offset=CameraCfg.OffsetCfg() if offset is None else offset,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=1.2,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 5.0),
            ),
        )
    )


def quat_wxyz_from_rpy_deg(rpy_deg: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Build a USD rotateXYZ-compatible quaternion from degrees."""
    rpy = torch.deg2rad(torch.tensor(rpy_deg, dtype=torch.float32))
    quat = quat_from_matrix(matrix_from_euler(rpy, "XYZ")).view(-1).cpu().numpy()
    return tuple(float(x) for x in quat)


def wrist_camera_offset(
    pos: tuple[float, float, float],
    quat_wxyz: tuple[float, float, float, float],
    rpy_deg: tuple[float, float, float] | None,
    convention: str,
) -> CameraCfg.OffsetCfg:
    rot = quat_wxyz if rpy_deg is None else quat_wxyz_from_rpy_deg(rpy_deg)
    return CameraCfg.OffsetCfg(
        pos=pos,
        rot=rot,
        convention=convention,
    )


def make_wrist_cameras(cfg: SceneBuildCfg) -> dict[str, Camera]:
    return {
        "left_wrist": make_rgb_camera(
            "/World/Robot/left_wrist_yaw_link/LeftWristCamera",
            cfg,
            offset=wrist_camera_offset(
                cfg.left_wrist_camera_pos,
                cfg.left_wrist_camera_quat_wxyz,
                cfg.left_wrist_camera_rpy_deg,
                cfg.wrist_camera_convention,
            ),
        ),
        "right_wrist": make_rgb_camera(
            "/World/Robot/right_wrist_yaw_link/RightWristCamera",
            cfg,
            offset=wrist_camera_offset(
                cfg.right_wrist_camera_pos,
                cfg.right_wrist_camera_quat_wxyz,
                cfg.right_wrist_camera_rpy_deg,
                cfg.wrist_camera_convention,
            ),
        ),
    }


def build_scene(cfg: SceneBuildCfg) -> dict[str, object]:
    spawn_background_and_table(cfg)
    print("[BOOT] spawning physics task objects...", flush=True)
    task_objects = spawn_physics_task_objects(cfg)
    camera = None
    wrist_cameras: dict[str, Camera] = {}
    if cfg.spawn_rgb_cameras:
        print("[BOOT] creating camera config...", flush=True)
        camera = make_rgb_camera("/World/DebugFrontCamera", cfg)
    print("[BOOT] creating robot articulation...", flush=True)
    robot = build_robot(
        "/World/Robot",
        cfg.joint_stiffness,
        cfg.joint_damping,
        cfg.joint_effort_limit,
        cfg.robot_base_z,
    )
    if cfg.spawn_rgb_cameras:
        print("[BOOT] creating wrist cameras...", flush=True)
        wrist_cameras = make_wrist_cameras(cfg)
    print("[BOOT] scene objects constructed.", flush=True)
    return {
        "robot": robot,
        "task_platform": task_objects["task_platform"],
        "red": task_objects["red"],
        "blue": task_objects["blue"],
        "plate": task_objects["plate"],
        "camera": camera,
        "wrist_cameras": wrist_cameras,
    }


def write_object_pose(
    obj: RigidObject,
    pos: np.ndarray,
    device: str,
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    pose = torch.tensor([[pos[0], pos[1], pos[2], *quat]], dtype=torch.float32, device=device)
    obj.write_root_pose_to_sim(pose)
    obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))


def reset_scene(scene: dict[str, object], cfg: SceneBuildCfg, sim: SimulationContext) -> np.ndarray:
    robot: Articulation = scene["robot"]

    default_drive = get_default_joint_positions()
    init_pos = torch.zeros(1, robot.num_joints, device=sim.device)
    for drive_i, joint_name in enumerate(ALL_DRIVE_JOINTS):
        if joint_name in robot.joint_names:
            init_pos[0, robot.joint_names.index(joint_name)] = float(default_drive[drive_i])
    robot.write_joint_state_to_sim(init_pos, torch.zeros_like(init_pos))
    robot.reset()

    write_object_pose(scene["task_platform"], cfg.layout.task_platform_pos(cfg.table_top_z), sim.device)
    write_object_pose(scene["red"], cfg.layout.red_block_pos(cfg.table_top_z), sim.device, PILL_BOTTLE_ORIENTATION)
    write_object_pose(scene["blue"], cfg.layout.blue_block_pos(cfg.table_top_z), sim.device)
    write_object_pose(scene["plate"], cfg.layout.plate_pos(cfg.table_top_z), sim.device)
    return init_pos[0].detach().cpu().numpy()


def reset_camera(camera: Camera, sim: SimulationContext, cfg: SceneBuildCfg | None = None) -> None:
    eye = cfg.camera_eye if cfg is not None else (0.10, 0.0, 1.80)
    target = cfg.camera_target if cfg is not None else (0.68, 0.0, 1.02)
    rpy_deg = cfg.camera_rpy_deg if cfg is not None else None
    if rpy_deg is None:
        camera.set_world_poses_from_view(
            eyes=torch.tensor([eye], dtype=torch.float32, device=sim.device),
            targets=torch.tensor([target], dtype=torch.float32, device=sim.device),
        )
    else:
        rpy = torch.deg2rad(torch.tensor(rpy_deg, dtype=torch.float32, device=sim.device))
        if cfg is not None and cfg.camera_convention == "opengl":
            # IsaacSim UI displays camera rotation as USD rotateXYZ. That is not
            # the same Euler decomposition as IsaacLab's quat_from_euler_xyz().
            # Build the quaternion from the rotateXYZ matrix so the UI fields
            # match camera_rpy_deg.
            quat = quat_from_matrix(matrix_from_euler(rpy, "XYZ")).view(1, 4)
        else:
            quat = quat_from_euler_xyz(rpy[0:1], rpy[1:2], rpy[2:3])
        camera.set_world_poses(
            positions=torch.tensor([eye], dtype=torch.float32, device=sim.device),
            orientations=quat,
            convention=cfg.camera_convention if cfg is not None else "opengl",
        )
    camera.reset()


def reset_viewport(sim: SimulationContext, cfg: SceneBuildCfg | None = None) -> None:
    """Set the Isaac Sim editor viewport without spawning RGB camera sensors."""
    eye = cfg.camera_eye if cfg is not None else (0.10, 0.0, 1.80)
    target = cfg.camera_target if cfg is not None else (0.68, 0.0, 1.02)
    sim.set_camera_view(list(eye), list(target))


def format_layout(cfg: SceneBuildCfg) -> str:
    red = cfg.layout.red_block_pos(cfg.table_top_z)
    blue = cfg.layout.blue_block_pos(cfg.table_top_z)
    plate = cfg.layout.plate_pos(cfg.table_top_z)
    task_platform = cfg.layout.task_platform_pos(cfg.table_top_z)
    return (
        "Task layout:\n"
        f"  robot_base_z={cfg.robot_base_z:.3f}\n"
        f"  task_surface_z={cfg.layout.task_surface_z(cfg.table_top_z):.3f}\n"
        f"  task_platform=({task_platform[0]:.3f}, {task_platform[1]:.3f}, {task_platform[2]:.3f}) "
        f"size=({TASK_PLATFORM_SIZE[0]:.3f}, {TASK_PLATFORM_SIZE[1]:.3f}, {TASK_PLATFORM_SIZE[2]:.3f})\n"
        f"  table_center=({cfg.layout.table_center_x:.3f}, {cfg.layout.table_center_y:.3f})\n"
        f"  red_bottle=({red[0]:.3f}, {red[1]:.3f}, {red[2]:.3f}) "
        f"asset={PILL_BOTTLE_USDZ.name} scale={PILL_BOTTLE_SCALE_VALUE:.6f} "
        f"approx_size=({PILL_BOTTLE_SIZE_M[0]:.3f},{PILL_BOTTLE_SIZE_M[1]:.3f},{PILL_BOTTLE_SIZE_M[2]:.3f})\n"
        f"  blue_block=({blue[0]:.3f}, {blue[1]:.3f}, {blue[2]:.3f})\n"
        f"  plate=({plate[0]:.3f}, {plate[1]:.3f}, {plate[2]:.3f})"
    )
