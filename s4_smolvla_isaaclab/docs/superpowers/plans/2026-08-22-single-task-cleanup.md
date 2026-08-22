# Drawer-Only Task Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the legacy blue-cylinder-to-plate task, its compatibility paths, documentation, assets, datasets, and checkpoints so the project exposes only `drawer_insert_close`.

**Architecture:** Keep the reusable robot, camera, IK, HDF5, LeRobotDataset, and rollout layers, but remove every runtime branch whose only consumer is `right_blue_cylinder_plate`. Convert the data path from a dual-mode `right_only|bimanual` API to the single 26D bimanual contract used by the drawer task, then verify the remaining task with pure CPU tests and static source checks.

**Tech Stack:** Bash, Python 3.11/3.12, pytest, HDF5/h5py, IsaacLab-facing Python modules (syntax/static checks only), Markdown.

**Spec:** `s4_smolvla_isaaclab/docs/superpowers/specs/2026-08-22-task-cleanup-periodic-train-eval-design.md`

## Global Constraints

- Do not modify `/home/zfy/smolVLA/lerobot` or `/home/zfy/IsaacLab`.
- Preserve `drawer_insert_close`, its 26D state/action order, absolute joint-target semantics, three cameras, 20 Hz dataset rate, and 120 Hz control interface.
- Preserve shared robot loading, dexterous hands, lighting, cameras, IK, gravity compensation, HDF5 writer infrastructure, conversion infrastructure, Policy Server, rollout, and diagnostics.
- Do not collect, convert, train, launch Isaac Sim, or load a GPU checkpoint while implementing this plan.
- Treat unrelated working-tree changes as user work; do not modify or revert them.
- Delete only the three explicitly approved historical output directories after resolving and checking their exact paths.
- Use TDD for behavior changes and make one focused commit per task.

---

## File Structure

### Delete

- `tasks/right_blue_cylinder_plate.py` — legacy task registration.
- `tasks/right_blue_cylinder_plate_controller.py` — legacy controller.
- `tasks/bimanual_red_blue_plate.py` — unused legacy task specification.
- `configs/tasks/right_blue_cylinder_plate.dataset.json` — legacy 13D dataset contract.
- `configs/tasks/right_blue_cylinder_plate.smolvla.yaml` — legacy training configuration.
- `scripts/pipeline_collect_convert_train.sh` — legacy cylinder/plate pipeline.
- `assets/scenes/Coca_Cola_Bottle.usdz` — unreferenced legacy scene asset.
- `assets/scenes/Pill_Bottle.usdz` — legacy red-object asset.
- `assets/scenes/task_sence.usd` — legacy cylinder/plate scene asset.

### Modify

- `tasks/__init__.py` — expose only the drawer task.
- `data/hdf5_schema.py` — remove legacy rigid-object field constants.
- `data/dataset_writer.py` — retain only the drawer object pose sequence.
- `data/lerobot_conversion.py` — make conversion unconditionally bimanual.
- `scripts/convert_lerobot.py` — remove `--control-mode` and legacy defaulting.
- `scripts/record_dataset.py` — remove legacy CLI flags, helpers, scene branch, and approximately 1,000-line cylinder state machine.
- `s4_robot/simulation.py` — retain shared simulation helpers; remove legacy task layout and scene builder.
- `tasks/drawer_insert_close_scene.py` — remove stale legacy prim cleanup names.
- `scripts/control_arm.py` — remove `reach-block` and `grasp-block`; retain generic hand, joint, TCP, reset, stop, and diagnostic commands.
- `scripts/preview_policy.py` — default to the active drawer dataset instead of a legacy repo ID.
- `scripts/visualize_policy.py` — default to the active drawer dataset instead of a legacy repo ID.
- `run.sh` — remove legacy block flags, `pipeline`, and `reach-block` aliases while preserving current drawer commands.
- `docs/knowledge_base/OPEN_ISSUES.md` — remove obsolete legacy-task issues.
- `docs/knowledge_base/PROBLEMS_AND_SOLUTIONS.md` — remove cylinder-only solutions.
- `docs/knowledge_base/PROJECT_STORY.md` — make the maintained project story drawer-only.
- `docs/knowledge_base/SMOLVLA_ISAACLAB_ROADMAP.md` — remove the legacy task history and obsolete commands.
- Relevant top-level docs discovered by the final source scan — remove only references proven to describe the deleted task.

### Create

- `tests/test_single_task_scope.py` — static single-task boundary and deleted-path contract.

---

### Task 1: Collapse the task registry to `drawer_insert_close`

**Files:**
- Modify: `tests/test_registry.py`
- Modify: `tasks/__init__.py`
- Delete: `tasks/right_blue_cylinder_plate.py`
- Delete: `tasks/right_blue_cylinder_plate_controller.py`
- Delete: `tasks/bimanual_red_blue_plate.py`
- Delete: `configs/tasks/right_blue_cylinder_plate.dataset.json`
- Delete: `configs/tasks/right_blue_cylinder_plate.smolvla.yaml`

**Interfaces:**
- Consumes: `TaskModuleSpec` and `TaskDataContract` from `tasks/base.py`.
- Produces: `TASK_REGISTRY == {"drawer_insert_close": DRAWER_INSERT_CLOSE}` and unchanged `get_task_spec(task_id: str) -> TaskModuleSpec`.

- [ ] **Step 1: Add the failing single-task registry test**

```python
from tasks import TASK_REGISTRY, get_task_spec


def test_only_drawer_task_is_registered():
    assert tuple(TASK_REGISTRY) == ("drawer_insert_close",)
    assert get_task_spec("drawer_insert_close") is TASK_REGISTRY["drawer_insert_close"]


def test_removed_task_is_rejected():
    import pytest

    with pytest.raises(KeyError, match="Available tasks: drawer_insert_close"):
        get_task_spec("right_blue_cylinder_plate")
```

- [ ] **Step 2: Run the test and verify the old task makes it fail**

Run:

```bash
python3 -m pytest tests/test_registry.py -v
```

Expected: `test_only_drawer_task_is_registered` fails because the registry still contains `right_blue_cylinder_plate`.

- [ ] **Step 3: Remove the legacy registry export and files**

Make `tasks/__init__.py` equivalent to:

```python
from .base import TaskDataContract, TaskModuleSpec
from .drawer_insert_close import TASK_SPEC as DRAWER_INSERT_CLOSE

TASK_REGISTRY = {DRAWER_INSERT_CLOSE.task_id: DRAWER_INSERT_CLOSE}


def get_task_spec(task_id: str) -> TaskModuleSpec:
    try:
        return TASK_REGISTRY[task_id]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_REGISTRY))
        raise KeyError(f"Unknown task_id={task_id!r}. Available tasks: {available}") from exc
```

Delete the five legacy implementation/configuration files listed above. Do not change `tasks/drawer_insert_close.py`, `tasks/base.py`, or the active drawer configurations.

- [ ] **Step 4: Run registry and configuration tests**

Run:

```bash
python3 -m pytest tests/test_registry.py tests/test_config.py tests/test_contract.py -v
```

Expected: all selected tests pass and the drawer task still reports the 26D `s4_bimanual_v1` contract.

- [ ] **Step 5: Commit the registry cleanup**

```bash
git add s4_smolvla_isaaclab/tasks s4_smolvla_isaaclab/configs/tasks s4_smolvla_isaaclab/tests/test_registry.py
git commit -m "refactor: keep only drawer task registration"
```

---

### Task 2: Remove the 13D `right_only` data compatibility contract

**Files:**
- Modify: `tests/test_hdf5_schema.py`
- Modify: `tests/test_contract.py`
- Modify: `data/hdf5_schema.py`
- Modify: `data/dataset_writer.py`
- Modify: `data/lerobot_conversion.py`
- Modify: `scripts/convert_lerobot.py`

**Interfaces:**
- Consumes: drawer HDF5 fields `processed_actions`, `obs/s4_active_joint_pos`, three RGB paths, per-frame language fields, and `drawer_task_object_pose`.
- Produces: `convert_hdf5_to_lerobot(..., language_contract=...) -> Path` with no `control_mode` parameter; converted state/action dimensions come directly from the 26D HDF5 arrays.

- [ ] **Step 1: Add failing tests for the drawer-only data API**

Add to `tests/test_contract.py`:

```python
import inspect

from data.lerobot_conversion import convert_hdf5_to_lerobot


def test_converter_has_no_legacy_control_mode():
    assert "control_mode" not in inspect.signature(convert_hdf5_to_lerobot).parameters
```

Add to `tests/test_hdf5_schema.py`:

```python
from data import hdf5_schema


def test_hdf5_schema_contains_only_current_task_object_pose():
    assert not hasattr(hdf5_schema, "RED_BLOCK_POSE")
    assert not hasattr(hdf5_schema, "BLUE_BLOCK_POSE")
    assert not hasattr(hdf5_schema, "PLATE_POSE")
    assert hdf5_schema.DRAWER_TASK_OBJECT_POSE.endswith("drawer_task_object/root_pose")
```

- [ ] **Step 2: Run the tests and verify they fail on legacy fields/mode**

Run:

```bash
python3 -m pytest tests/test_contract.py tests/test_hdf5_schema.py -v
```

Expected: failures show that `control_mode` and the three legacy pose constants still exist.

- [ ] **Step 3: Remove legacy HDF5 buffer fields**

In `data/hdf5_schema.py`, remove `RED_BLOCK_POSE`, `BLUE_BLOCK_POSE`, and `PLATE_POSE`.

In `EpisodeBuffer`, remove:

```python
red_block_pose
blue_block_pose
plate_pose
```

Remove them from validation and `Hdf5DemoWriter.write_episode()`. Preserve `drawer_task_object_pose` and all current language/camera/state/action fields.

- [ ] **Step 4: Make LeRobot conversion always bimanual**

Remove `RIGHT_ONLY_ACTION_SLICE`, both `right_only` branches, the mode validator, and the `control_mode` argument. `inspect_first_demo()` must always infer dimensions from `processed_actions` and `obs/s4_active_joint_pos` when present, falling back to full joint positions only as it does now.

The frame loop must remain:

```python
actions = np.asarray(demo[schema.PROCESSED_ACTIONS])
state_path = schema.ACTIVE_JOINT_POS if schema.ACTIVE_JOINT_POS in demo else schema.FULL_JOINT_POS
states = np.asarray(demo[state_path])
```

In `scripts/convert_lerobot.py`, remove `--control-mode`, do not read `dataset.control_mode`, and do not pass it to the converter. Keep all FPS, scene-contract, language-contract, camera, overwrite, and video behavior unchanged.

- [ ] **Step 5: Run data-contract tests**

Run:

```bash
python3 -m pytest tests/test_hdf5_schema.py tests/test_contract.py tests/test_language_phases.py tests/test_video.py -v
```

Expected: all selected tests pass; the HDF5 fixture remains 26D and the language conversion tests are unchanged.

- [ ] **Step 6: Commit the data cleanup**

```bash
git add s4_smolvla_isaaclab/data s4_smolvla_isaaclab/scripts/convert_lerobot.py s4_smolvla_isaaclab/tests
git commit -m "refactor: remove right-only dataset compatibility"
```

---

### Task 3: Remove the legacy recorder, scene, pipeline, and manual-control paths

**Files:**
- Create: `tests/test_single_task_scope.py`
- Modify: `scripts/record_dataset.py`
- Modify: `s4_robot/simulation.py`
- Modify: `tasks/drawer_insert_close_scene.py`
- Modify: `scripts/control_arm.py`
- Modify: `scripts/preview_policy.py`
- Modify: `scripts/visualize_policy.py`
- Modify: `run.sh`
- Delete: `scripts/pipeline_collect_convert_train.sh`
- Delete: `assets/scenes/Coca_Cola_Bottle.usdz`
- Delete: `assets/scenes/Pill_Bottle.usdz`
- Delete: `assets/scenes/task_sence.usd`

**Interfaces:**
- Consumes: `tasks.drawer_insert_close_scene:build_scene`, `run_static_task_scene()`, `SceneBuildCfg`, `create_simulation_context()`, camera helpers, `write_object_pose()`, and generic manual-control JSON messages.
- Produces: drawer-only `record_dataset.py`; shared `s4_robot.simulation` without a default task scene; unchanged `run.sh record`, `sim`, `collect-convert`, `collect-train`, `train`, and `rollout` commands.

- [ ] **Step 1: Write the failing static scope test**

Create `tests/test_single_task_scope.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_TOKENS = (
    "right_blue_cylinder_plate",
    "s4_right_blue_cylinder_plate",
    "RIGHT_ONLY_ACTION_SLICE",
    "final_cylinder_in_plate",
    "randomize_blue_xy",
)


def test_runtime_sources_do_not_contain_legacy_task_contract():
    paths = [
        ROOT / "run.sh",
        ROOT / "scripts/record_dataset.py",
        ROOT / "scripts/convert_lerobot.py",
        ROOT / "s4_robot/simulation.py",
        ROOT / "tasks/__init__.py",
        ROOT / "data/lerobot_conversion.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    for token in LEGACY_TOKENS:
        assert token not in combined


def test_legacy_task_files_and_assets_are_removed():
    removed = [
        "scripts/pipeline_collect_convert_train.sh",
        "assets/scenes/Coca_Cola_Bottle.usdz",
        "assets/scenes/Pill_Bottle.usdz",
        "assets/scenes/task_sence.usd",
    ]
    assert all(not (ROOT / relative).exists() for relative in removed)
```

- [ ] **Step 2: Run the static scope test and verify it fails**

Run:

```bash
python3 -m pytest tests/test_single_task_scope.py -v
```

Expected: both tests fail because the runtime branch, old pipeline, and assets still exist.

- [ ] **Step 3: Reduce `record_dataset.py` to the drawer execution path**

Remove cylinder-only CLI arguments (`task-x`, `task-y`, `block-y-offset`, `plate-x`, `auto-grasp-block`, blue XY flags, plate success flags, and legacy reach/grasp flags), imports, helpers, pose recording, and control-file payload branches.

Delete the cylinder-only `run_debug()` body beginning with its `right_blue_cylinder_plate` branch. Rename the remaining dispatch clearly or call the existing drawer runner directly:

```python
print("[BOOT] entering run loop...", flush=True)
run_static_task_scene(scene, cfg, sim)
```

Keep all drawer collection behavior, debug visualization, gravity compensation, failure reporting, retry policy, resumption, cameras, and transactional HDF5 writes unchanged.

Simplify `resolve_scene_builder()` to import the active registered drawer builder without the `s4_robot.simulation:build_scene` compatibility special case. `load_active_drawer_scripted_cfg()` may become an unconditional active-task scripted-config loader because only one task remains.

- [ ] **Step 4: Reduce `s4_robot/simulation.py` to shared infrastructure**

Remove old object constants, `TASK_OBJECT_KEYS`, `TaskLayout`, `spawn_physics_task_objects()`, the default `build_scene()`, `reset_scene()`, and `format_layout()`. Remove the `layout` field from `SceneBuildCfg` and update recorder construction accordingly.

Retain these shared interfaces because drawer code imports them:

```text
SceneBuildCfg
create_simulation_context
build_robot
apply_finger_visual_material
spawn_background_and_table
configure_fixed_lighting
configure_usdz_rigid_meshes
make_rgb_camera
make_wrist_cameras
write_object_pose
reset_camera
```

Remove `/World/TaskPlatform`, `/World/RedBlock`, `/World/BlueBlock`, and `/World/Plate` from `tasks/drawer_insert_close_scene.py::_remove_stale_task_prims()` after the old scene builder is gone.

- [ ] **Step 5: Remove old control and pipeline entry points**

Delete `scripts/pipeline_collect_convert_train.sh` and remove the `pipeline` case/help alias from `run.sh`.

Remove the `reach-block` and `grasp-block` subparsers and their payload branches from `scripts/control_arm.py`; retain `hand`, direct joint tests, `tcp-pose`, `stop`, `reset-scene`, and diagnostics. Remove the top-level `reach-block` alias from `run.sh` but retain `control`.

In `record_command()`, remove the local `block` variable and stop passing `--auto-grasp-block`; keep `--auto-grasp` because it starts the drawer scripted collector.

Change preview/visualization defaults to the active project configuration rather than the string `s4_right_blue_cylinder_plate_v1`. Use `load_project_config().dataset.repo_id.split("/")[-1]` or make the CLI default `None` and resolve it in `main()`.

- [ ] **Step 6: Delete the proven old-only scene assets**

Delete the three files listed above only after this command returns no references outside the files being removed:

```bash
rg -n "Coca_Cola_Bottle|Pill_Bottle|task_sence" . --glob '!docs/superpowers/**' --glob '!**/.git/**'
```

Expected before deletion: `Pill_Bottle` is referenced only by the old scene code; the other two assets have no maintained runtime consumer. Expected after code cleanup: no results.

- [ ] **Step 7: Run syntax and scope checks**

Run:

```bash
bash -n run.sh scripts/collect_convert.sh scripts/collect_convert_check_train.sh scripts/train_smolvla_local.sh
python3 -m pytest tests/test_single_task_scope.py tests/test_registry.py tests/test_contract.py -v
python3 - <<'PY'
import ast
from pathlib import Path

for relative in (
    "scripts/record_dataset.py",
    "scripts/control_arm.py",
    "scripts/preview_policy.py",
    "scripts/visualize_policy.py",
    "s4_robot/simulation.py",
    "tasks/drawer_insert_close_scene.py",
):
    ast.parse(Path(relative).read_text(encoding="utf-8"), filename=relative)
print("AST parse passed")
PY
```

Expected: Bash parsing, AST parsing, and selected tests pass without launching Isaac Sim.

- [ ] **Step 8: Commit the runtime cleanup**

```bash
git add s4_smolvla_isaaclab/run.sh s4_smolvla_isaaclab/scripts s4_smolvla_isaaclab/s4_robot s4_smolvla_isaaclab/tasks/drawer_insert_close_scene.py s4_smolvla_isaaclab/assets/scenes s4_smolvla_isaaclab/tests/test_single_task_scope.py
git commit -m "refactor: remove legacy cylinder runtime"
```

---

### Task 4: Remove legacy task documentation and update command references

**Files:**
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `docs/QUICKSTART.md`
- Modify: `docs/PROJECT_STRUCTURE.md`
- Modify: `docs/TASK_SYSTEM.md`
- Modify: `docs/knowledge_base/OPEN_ISSUES.md`
- Modify: `docs/knowledge_base/PROBLEMS_AND_SOLUTIONS.md`
- Modify: `docs/knowledge_base/PROJECT_STORY.md`
- Modify: `docs/knowledge_base/SMOLVLA_ISAACLAB_ROADMAP.md`
- Modify: any additional maintained Markdown file returned by the source scan.
- Modify: `tests/test_single_task_scope.py`

**Interfaces:**
- Consumes: current `bash run.sh help`, drawer configuration names, and current dataset/checkpoint paths.
- Produces: documentation that presents `drawer_insert_close` as the only supported task and contains no executable legacy command.

- [ ] **Step 1: Extend the failing scope test to maintained documentation**

Add:

```python
def test_maintained_docs_do_not_advertise_legacy_task():
    docs = [
        ROOT / "README.md",
        *(ROOT / "docs").glob("*.md"),
        *(ROOT / "docs/knowledge_base").glob("*.md"),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in docs)
    for token in ("right_blue_cylinder_plate", "s4_right_blue", "--block blue"):
        assert token not in combined
```

Exclude `docs/superpowers/specs/` and `docs/superpowers/plans/`, which intentionally record the removal decision.

- [ ] **Step 2: Run the documentation boundary test and verify it fails**

Run:

```bash
python3 -m pytest tests/test_single_task_scope.py::test_maintained_docs_do_not_advertise_legacy_task -v
```

Expected: failure identifies the remaining historical knowledge-base references.

- [ ] **Step 3: Rewrite the maintained documentation around the drawer-only project**

Remove legacy commands, task IDs, 13D/right-only schema explanations, cylinder/plate success criteria, and old output paths. In historical files, preserve lessons that still apply to the drawer task only after rewriting them without the deleted task’s objects or commands. Do not remove current drawer grasp diagnostics or the new 10-language-phase material.

Update command lists from the actual `run.sh` help output. Do not introduce `train-eval` here; that interface is delivered by the second plan.

- [ ] **Step 4: Verify documentation and help**

Run:

```bash
python3 -m pytest tests/test_single_task_scope.py -v
bash run.sh help
bash run.sh list-tasks
rg -n "right_blue_cylinder|s4_right_blue|s4_right_v1|right_only|--block blue" README.md docs run.sh scripts tasks data configs --glob '!docs/superpowers/**'
```

Expected: tests pass; `list-tasks` reports only `drawer_insert_close`; the final `rg` has no results.

- [ ] **Step 5: Commit the documentation cleanup**

```bash
git add s4_smolvla_isaaclab/README.md s4_smolvla_isaaclab/docs s4_smolvla_isaaclab/tests/test_single_task_scope.py
git commit -m "docs: remove legacy cylinder task guidance"
```

---

### Task 5: Delete the explicitly approved historical outputs

**Files:**
- Delete: `datasets/staging/s4_right_blue_cylinder_plate_v1/`
- Delete: `datasets/lerobot_data/s4_right_blue_cylinder_plate_v1/`
- Delete: `outputs/train/smolvla_s4_right_v1/`

**Interfaces:**
- Consumes: exact project root `/home/zfy/smolVLA/s4_smolvla_isaaclab`.
- Produces: no legacy cylinder HDF5, LeRobotDataset, or checkpoint directory; current drawer datasets/checkpoints remain untouched.

- [ ] **Step 1: Resolve and display exact deletion targets**

Run:

```bash
realpath /home/zfy/smolVLA/s4_smolvla_isaaclab/datasets/staging/s4_right_blue_cylinder_plate_v1
realpath /home/zfy/smolVLA/s4_smolvla_isaaclab/datasets/lerobot_data/s4_right_blue_cylinder_plate_v1
realpath /home/zfy/smolVLA/s4_smolvla_isaaclab/outputs/train/smolvla_s4_right_v1
du -sh /home/zfy/smolVLA/s4_smolvla_isaaclab/datasets/staging/s4_right_blue_cylinder_plate_v1 /home/zfy/smolVLA/s4_smolvla_isaaclab/datasets/lerobot_data/s4_right_blue_cylinder_plate_v1 /home/zfy/smolVLA/s4_smolvla_isaaclab/outputs/train/smolvla_s4_right_v1
```

Expected: all resolved paths remain below `/home/zfy/smolVLA/s4_smolvla_isaaclab/` and match the three names exactly. If any path resolves elsewhere, stop.

- [ ] **Step 2: Confirm current drawer artifacts are different paths**

Run:

```bash
find /home/zfy/smolVLA/s4_smolvla_isaaclab/datasets -maxdepth 2 -mindepth 2 -type d -printf '%p\n' | sort
find /home/zfy/smolVLA/s4_smolvla_isaaclab/outputs/train -maxdepth 1 -mindepth 1 -type d -printf '%p\n' | sort
```

Expected: drawer paths contain `drawer_insert_close`; none of them are included in the deletion targets.

- [ ] **Step 3: Delete exactly the approved directories**

Delete only the three explicit absolute paths. Do not use a wildcard, environment variable, project root, dataset root, or output root as a recursive deletion target.

- [ ] **Step 4: Verify deletion and report recoverability**

Run:

```bash
test ! -e /home/zfy/smolVLA/s4_smolvla_isaaclab/datasets/staging/s4_right_blue_cylinder_plate_v1
test ! -e /home/zfy/smolVLA/s4_smolvla_isaaclab/datasets/lerobot_data/s4_right_blue_cylinder_plate_v1
test ! -e /home/zfy/smolVLA/s4_smolvla_isaaclab/outputs/train/smolvla_s4_right_v1
```

Expected: all commands return zero. Report the measured deleted size and state that recovery requires an external backup.

---

### Task 6: Run the drawer-only regression gate

**Files:**
- Modify only if a verified regression requires a narrowly scoped correction.

**Interfaces:**
- Consumes: all cleanup tasks.
- Produces: evidence that the maintained CPU contracts and command routing remain valid without touching LeRobot or launching GPU workloads.

- [ ] **Step 1: Verify the working-tree diff and deleted-path scope**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Expected: no whitespace errors; only intended project files appear.

- [ ] **Step 2: Run the complete pure test suite**

Run:

```bash
python3 -m pytest tests -v
```

Expected: all pure tests pass. If environment-specific tests are marked/skipped by their existing conditions, report them separately and do not claim they ran.

- [ ] **Step 3: Re-run command and source-contract checks**

Run:

```bash
bash -n run.sh scripts/*.sh
bash run.sh help
bash run.sh list-tasks
rg -n "right_blue_cylinder|s4_right_blue|s4_right_v1|right_only|red_block_pose|blue_block_pose|plate_pose" README.md docs run.sh scripts tasks data configs s4_robot --glob '!docs/superpowers/**'
```

Expected: Shell syntax passes, only the drawer task is listed, and `rg` returns no maintained-code/document matches.

- [ ] **Step 4: Verify repository boundaries**

Run:

```bash
git -C /home/zfy/smolVLA/lerobot status --short
git -C /home/zfy/IsaacLab status --short
```

Expected: this implementation introduced no changes in either external repository. Existing unrelated changes, if any, must be reported rather than modified.

- [ ] **Step 5: Commit any final narrow regression correction**

If Step 2 or Step 3 required a correction, stage only those files and commit:

```bash
git commit -m "test: verify drawer-only project contracts"
```

If no correction was needed, do not create an empty commit.
