# Periodic Train-Eval Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe `bash run.sh train-eval` workflow that trains in 50,000-step segments, evaluates every saved checkpoint on the same 10 randomized drawer scenes, validates all artifacts, and resumes until the configured final step.

**Architecture:** Implement deterministic scenario-manifest and workflow-validation logic as pure functions in `s4_pipeline/periodic_eval.py`. Extend the existing Rollout entry point only to replay a manifest and enrich diagnostics, then add a CPU-light orchestrator that starts training and Rollout as sequential child processes, streams their output to terminal and logs, and atomically records workflow progress.

**Tech Stack:** Python 3.11/3.12, Bash, pytest, JSON/CSV, subprocess, hashlib, IsaacLab Rollout CLI, LeRobot training CLI through the existing wrapper.

**Spec:** `s4_smolvla_isaaclab/docs/superpowers/specs/2026-08-22-task-cleanup-periodic-train-eval-design.md`

## Global Constraints

- Do not modify `/home/zfy/smolVLA/lerobot` or `/home/zfy/IsaacLab`.
- Keep `bash run.sh train` as pure training with unchanged behavior.
- Default interval is 50,000 steps and default periodic evaluation is 10 episodes.
- Evaluate the final target even when total steps are not divisible by 50,000.
- Training must exit normally before Rollout starts; never pause, kill, or overlap GPU training and evaluation.
- Reuse the same exact 10 randomized scenarios for every checkpoint.
- Drawer initial opening is randomized from the active scripted configuration, currently `[0.00, 0.05] m`.
- Keep current Rollout control behavior: 50-frame chunks, 40-frame replanning, 5-frame overlap blend, 8-frame phase blend, 20-frame normal extension, and 80-frame drawer approach/pull extension unless explicitly overridden by CLI.
- Task failure is an evaluation result and does not stop training; infrastructure failure or incomplete artifacts stops the workflow.
- Do not automatically launch real training, Isaac Sim, Policy Server, or GPU inference during implementation verification.

---

## File Structure

### Create

- `s4_pipeline/periodic_eval.py` — pure node planning, scenario manifests, fingerprints, checkpoint validation, artifact validation, atomic state, and command construction.
- `scripts/train_eval.py` — sequential orchestration, streamed logging, resume decisions, and exit status.
- `tests/test_periodic_eval.py` — pure CPU contract tests.

### Modify

- `scripts/eval_policy.py` — accept/replay a scenario manifest and record phase/forced-transition diagnostics.
- `s4_pipeline/rollout_metrics.py` — include failure reasons and transition diagnostics in summaries without changing control behavior.
- `tests/test_rollout_metrics.py` — verify new diagnostic aggregation.
- `run.sh` — expose `train-eval` while preserving `train`.
- `docs/TRAINING.md` — document segmented training and recovery.
- `docs/ONLINE_ROLLOUT.md` — document fixed randomized scenario replay.
- `docs/QUICKSTART.md` — add a concise periodic workflow command.
- `docs/README.md` — link the workflow documentation.
- `docs/course/03_PROJECT_DEPLOYMENT.md` — add deployment-facing use and output layout.

---

### Task 1: Implement evaluation-node planning and atomic workflow state

**Files:**
- Create: `tests/test_periodic_eval.py`
- Create: `s4_pipeline/periodic_eval.py`

**Interfaces:**
- Consumes: total step, interval, filesystem paths, and JSON-compatible dictionaries.
- Produces:
  - `evaluation_steps(total_steps: int, interval: int) -> list[int]`
  - `canonical_fingerprint(payload: dict[str, Any]) -> str`
  - `atomic_write_json(path: Path, payload: dict[str, Any]) -> Path`
  - `load_workflow_state(path: Path) -> dict[str, Any] | None`
  - `validate_workflow_state(state: dict[str, Any], expected_fingerprint: str) -> None`

- [ ] **Step 1: Write failing node-planning and atomic-state tests**

Create these tests:

```python
import json
from pathlib import Path

import pytest

from s4_pipeline.periodic_eval import (
    atomic_write_json,
    canonical_fingerprint,
    evaluation_steps,
    load_workflow_state,
    validate_workflow_state,
)


def test_evaluation_steps_include_partial_final_target():
    assert evaluation_steps(180_000, 50_000) == [50_000, 100_000, 150_000, 180_000]
    assert evaluation_steps(500_000, 50_000)[-1] == 500_000


def test_evaluation_steps_reject_invalid_values():
    with pytest.raises(ValueError, match="total_steps"):
        evaluation_steps(0, 50_000)
    with pytest.raises(ValueError, match="interval"):
        evaluation_steps(100_000, 0)


def test_atomic_state_round_trip_and_fingerprint_guard(tmp_path):
    path = tmp_path / "workflow_state.json"
    fingerprint = canonical_fingerprint({"steps": 180_000, "interval": 50_000})
    state = {"workflow_fingerprint": fingerprint, "completed_evaluations": [50_000]}
    atomic_write_json(path, state)
    assert load_workflow_state(path) == state
    validate_workflow_state(state, fingerprint)
    with pytest.raises(ValueError, match="fingerprint"):
        validate_workflow_state(state, "different")
    assert not path.with_suffix(path.suffix + ".tmp").exists()
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run:

```bash
python3 -m pytest tests/test_periodic_eval.py -v
```

Expected: collection fails with `ModuleNotFoundError: s4_pipeline.periodic_eval`.

- [ ] **Step 3: Implement minimal pure helpers**

Use deterministic JSON encoding for fingerprints:

```python
def canonical_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

`evaluation_steps()` must generate interval multiples below the target and append the target exactly once. `atomic_write_json()` must create the parent, write `<name>.tmp`, flush and `os.fsync()`, then call `os.replace()`.

`validate_workflow_state()` must reject a missing or unequal `workflow_fingerprint`; it must not silently reset incompatible state.

- [ ] **Step 4: Run the new test file**

Run:

```bash
python3 -m pytest tests/test_periodic_eval.py -v
```

Expected: all initial periodic-eval tests pass.

- [ ] **Step 5: Commit the workflow primitives**

```bash
git add s4_smolvla_isaaclab/s4_pipeline/periodic_eval.py s4_smolvla_isaaclab/tests/test_periodic_eval.py
git commit -m "feat: add periodic evaluation workflow primitives"
```

---

### Task 2: Implement and validate the fixed randomized scenario manifest

**Files:**
- Modify: `tests/test_periodic_eval.py`
- Modify: `s4_pipeline/periodic_eval.py`

**Interfaces:**
- Consumes: active task ID, resolved randomization configuration, dataset/scene contract dictionary, episode count, seed, and the existing randomization sampler.
- Produces:
  - `scenario_contract_payload(task_id: str, randomization_cfg: dict[str, Any], dataset_contract: dict[str, Any]) -> dict[str, Any]`
  - `build_scenario_manifest(..., episode_count: int, seed: int) -> dict[str, Any]`
  - `load_or_create_scenario_manifest(path: Path, ..., episode_count: int, seed: int) -> dict[str, Any]`
  - `scenario_samples(manifest: dict[str, Any], expected_fingerprint: str, expected_count: int) -> list[dict[str, Any]]`

- [ ] **Step 1: Add failing deterministic-manifest tests**

```python
from s4_pipeline.periodic_eval import (
    build_scenario_manifest,
    canonical_fingerprint,
    scenario_contract_payload,
    scenario_samples,
)


RANDOMIZATION = {
    "enabled": True,
    "can_xy": {
        "enabled": True,
        "sampling": "stratified_grid",
        "grid_cells": [5, 5],
        "x_range": [-0.025, -0.015],
        "y_range": [-0.06, 0.01],
    },
    "drawer_initial_open": {"enabled": True, "range": [0.0, 0.05]},
    "distractor_cans_enabled": False,
}


def test_manifest_replays_identical_random_scenarios():
    contract = {"schema_version": "s4_bimanual_v1", "fps": 20}
    first = build_scenario_manifest(
        task_id="drawer_insert_close",
        randomization_cfg=RANDOMIZATION,
        dataset_contract=contract,
        episode_count=10,
        seed=42,
    )
    second = build_scenario_manifest(
        task_id="drawer_insert_close",
        randomization_cfg=RANDOMIZATION,
        dataset_contract=contract,
        episode_count=10,
        seed=42,
    )
    assert first == second
    assert len(first["scenarios"]) == 10
    assert all(0.0 <= row["sample"]["drawer_open_m"] <= 0.05 for row in first["scenarios"])
    assert len({tuple(row["sample"]["can_grid_cell"]) for row in first["scenarios"]}) == 10


def test_manifest_rejects_contract_or_count_drift():
    manifest = build_scenario_manifest(
        task_id="drawer_insert_close",
        randomization_cfg=RANDOMIZATION,
        dataset_contract={"schema_version": "s4_bimanual_v1"},
        episode_count=10,
        seed=42,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        scenario_samples(manifest, expected_fingerprint="changed", expected_count=10)
    with pytest.raises(ValueError, match="scenario count"):
        scenario_samples(manifest, expected_fingerprint=manifest["contract_fingerprint"], expected_count=9)
```

- [ ] **Step 2: Run the tests and verify new functions are missing**

Run:

```bash
python3 -m pytest tests/test_periodic_eval.py -v
```

Expected: import errors identify the manifest interfaces.

- [ ] **Step 3: Implement manifest generation using the production sampler**

Import and use existing helpers rather than reimplementing randomization:

```python
rng = make_randomization_rng(seed)
grid = make_can_grid_sampler(randomization_cfg, rng)
samples = [
    sample_randomization(randomization_cfg, seed=seed, rng=rng, can_grid_sampler=grid)
    for _ in range(episode_count)
]
```

Manifest schema:

```python
{
    "manifest_version": 1,
    "task_id": task_id,
    "seed": seed,
    "episode_count": episode_count,
    "contract_fingerprint": canonical_fingerprint(contract_payload),
    "contract": contract_payload,
    "scenarios": [
        {"scenario_index": index, "sample": sample}
        for index, sample in enumerate(samples)
    ],
}
```

The contract payload must include task ID, randomization settings, schema version, FPS, cameras, distractor contract, grasp-can nominal position/scale, and language-contract version when present. Existing manifests are read and validated; they are never regenerated silently.

- [ ] **Step 4: Run manifest and existing randomization tests**

Run:

```bash
python3 -m pytest tests/test_periodic_eval.py tests/test_randomization.py tests/test_rollout_metrics.py -v
```

Expected: all selected tests pass and current randomization behavior is unchanged.

- [ ] **Step 5: Commit the manifest contract**

```bash
git add s4_smolvla_isaaclab/s4_pipeline/periodic_eval.py s4_smolvla_isaaclab/tests/test_periodic_eval.py
git commit -m "feat: add reproducible rollout scenario manifests"
```

---

### Task 3: Add manifest replay and richer non-controlling Rollout diagnostics

**Files:**
- Modify: `scripts/eval_policy.py`
- Modify: `s4_pipeline/rollout_metrics.py`
- Modify: `tests/test_rollout_metrics.py`
- Modify: `tests/test_periodic_eval.py`

**Interfaces:**
- Consumes: optional `--scenario-manifest PATH`; existing randomization config and dataset contract.
- Produces: exact scenario replay; summary episode fields `scenario_index`, `final_phase_index`, `final_phase_id`, `forced_transition_count`, `forced_transitions`, and `failure_reasons`.

- [ ] **Step 1: Add failing diagnostic aggregation tests**

```python
from s4_pipeline.rollout_metrics import rollout_failure_reasons


def test_rollout_failure_reasons_distinguish_completion_and_task_checks():
    assert rollout_failure_reasons(complete=True, drawer_ok=True, can_ok=True) == []
    assert rollout_failure_reasons(complete=False, drawer_ok=False, can_ok=True) == [
        "phase_schedule_incomplete",
        "drawer_not_closed",
    ]
    assert rollout_failure_reasons(complete=True, drawer_ok=True, can_ok=False) == ["can_not_in_drawer"]
```

Add a manifest-replay test that writes a 10-scenario manifest, reloads it, and asserts the returned samples equal the original list byte-for-byte after JSON round-trip.

- [ ] **Step 2: Run tests and verify failure on missing diagnostic helper**

Run:

```bash
python3 -m pytest tests/test_rollout_metrics.py tests/test_periodic_eval.py -v
```

Expected: failure because `rollout_failure_reasons` is not implemented.

- [ ] **Step 3: Add `--scenario-manifest` without changing default rollout**

Add:

```python
parser.add_argument(
    "--scenario-manifest",
    type=Path,
    default=None,
    help="Create or replay an exact periodic-evaluation scenario manifest.",
)
```

After resolving `random_cfg`, dataset contract, grasp-can settings, and distractor settings, build the scenario contract. If a manifest path is supplied, call `load_or_create_scenario_manifest()` and require `--episodes` to equal its scenario count. In the episode loop use:

```python
if manifest_samples is None:
    init_sample = sample_randomization(...)
else:
    init_sample = dict(manifest_samples[episode_index])
```

When no manifest is provided, preserve the current shared-RNG and stratified-grid behavior exactly.

- [ ] **Step 4: Record phase and forced-transition evidence only**

Track forced transitions when the existing gate extension is exhausted, but do not change gate decisions or motion:

```python
forced_transitions.append({
    "phase_index": phase_index,
    "language_phase_id": schedule[phase_index].get("language_phase_id"),
    "extension_frames": phase_extension,
    "reasons": list(gate_reasons),
})
```

At episode completion add the new fields and compute `failure_reasons` from `complete`, `drawer_ok`, and `can_ok`. Include the manifest fingerprint at aggregate summary level. These fields are diagnostic only and must not trigger action correction or task retries.

- [ ] **Step 5: Run rollout pure tests and syntax parsing**

Run:

```bash
python3 -m pytest tests/test_rollout_metrics.py tests/test_periodic_eval.py tests/test_policy_protocol.py -v
python3 - <<'PY'
import ast
from pathlib import Path

ast.parse(Path("scripts/eval_policy.py").read_text(encoding="utf-8"), filename="scripts/eval_policy.py")
print("eval_policy AST parse passed")
PY
```

Expected: all pure tests and AST parsing pass; Isaac Sim is not started.

- [ ] **Step 6: Commit manifest replay**

```bash
git add s4_smolvla_isaaclab/scripts/eval_policy.py s4_smolvla_isaaclab/s4_pipeline/rollout_metrics.py s4_smolvla_isaaclab/tests
git commit -m "feat: replay fixed scenarios during rollout"
```

---

### Task 4: Validate resumable checkpoints and periodic evaluation artifacts

**Files:**
- Modify: `s4_pipeline/periodic_eval.py`
- Modify: `tests/test_periodic_eval.py`

**Interfaces:**
- Consumes: training output directory, expected step, step evaluation directory, episode count, and expected manifest fingerprint.
- Produces:
  - `checkpoint_dir_for_step(output_dir: Path, step: int) -> Path`
  - `validate_resumable_checkpoint(output_dir: Path, expected_step: int) -> Path`
  - `validate_rollout_artifacts(run_dir: Path, expected_episodes: int, manifest_fingerprint: str) -> dict[str, Any]`
  - `write_summary_csv(summary: dict[str, Any], path: Path) -> Path`

- [ ] **Step 1: Add failing checkpoint validation tests**

```python
def make_checkpoint(root, step):
    checkpoint = root / "checkpoints" / f"{step:06d}"
    files = {
        "pretrained_model/config.json": "{}",
        "pretrained_model/model.safetensors": "weights",
        "pretrained_model/train_config.json": "{}",
        "training_state/training_step.json": json.dumps({"step": step}),
        "training_state/optimizer_param_groups.json": "{}",
        "training_state/optimizer_state.safetensors": "optimizer",
        "training_state/rng_state.safetensors": "rng",
        "training_state/scheduler_state.json": "{}",
    }
    for relative, content in files.items():
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return checkpoint


def test_checkpoint_validation_requires_full_training_state(tmp_path):
    checkpoint = make_checkpoint(tmp_path, 50_000)
    assert validate_resumable_checkpoint(tmp_path, 50_000) == checkpoint / "pretrained_model"
    (checkpoint / "training_state/scheduler_state.json").unlink()
    with pytest.raises(ValueError, match="scheduler_state.json"):
        validate_resumable_checkpoint(tmp_path, 50_000)
```

- [ ] **Step 2: Add failing artifact validation tests**

Construct a `summary.json` with exactly 10 `episodes_detail` rows. For each row, create non-empty `ep001.avi`, `ep001_actions.csv`, and `ep001_actions.png` through `ep010.*`. Assert validation passes, then delete one PNG and assert a `ValueError` names the missing file. Assert a 9-episode summary is rejected even if the command exited zero.

- [ ] **Step 3: Run tests and verify the validation functions are missing**

Run:

```bash
python3 -m pytest tests/test_periodic_eval.py -v
```

Expected: failures identify the checkpoint/artifact interfaces.

- [ ] **Step 4: Implement strict validators and CSV output**

Checkpoint validation must require the eight fixture paths above, reject zero-byte binary state files, parse `training_step.json`, and require its `step` to equal `expected_step`.

Artifact validation must parse `summary.json`, require exactly 10 unique episode numbers, check `video`, `diagnostics_csv`, and `diagnostics_plot` paths from each result, ensure files exist and are non-empty, and require the summary’s manifest fingerprint to match. It must not require `success=True` or `complete=True`.

`write_summary_csv()` writes one row per episode with checkpoint step, scenario index, complete, success, failure reasons, drawer result, can result, final phase, forced transition count, initial randomization, and artifact paths.

- [ ] **Step 5: Run validation tests**

Run:

```bash
python3 -m pytest tests/test_periodic_eval.py -v
```

Expected: complete fixtures pass; incomplete checkpoint and evaluation fixtures fail with specific paths.

- [ ] **Step 6: Commit validators**

```bash
git add s4_smolvla_isaaclab/s4_pipeline/periodic_eval.py s4_smolvla_isaaclab/tests/test_periodic_eval.py
git commit -m "feat: validate periodic checkpoints and rollout artifacts"
```

---

### Task 5: Add the `train-eval` orchestrator and dry-run interface

**Files:**
- Create: `scripts/train_eval.py`
- Modify: `s4_pipeline/periodic_eval.py`
- Modify: `tests/test_periodic_eval.py`
- Modify: `run.sh`

**Interfaces:**
- Consumes: training config path, `--steps`, `--eval-interval`, `--eval-episodes`, `--seed`, `--resume`, `--overwrite-output`, `--dry-run`, and pass-through Rollout timing options.
- Produces: `bash run.sh train-eval`; streamed logs; `periodic_eval/scenario_manifest.json`, `workflow_state.json`, `train_eval.log`, `step_NNNNNN/`, and `summary.csv`.

- [ ] **Step 1: Add failing command-construction tests**

Add pure interfaces to the imports and tests:

```python
def test_train_and_rollout_commands_are_segmented_and_sequential(tmp_path):
    train = build_train_command(
        project_root=tmp_path,
        config=Path("configs/task.yaml"),
        target_step=100_000,
        save_freq=50_000,
        resume=True,
    )
    assert train == [
        "bash", str(tmp_path / "run.sh"), "train", "configs/task.yaml",
        "--steps", "100000", "--save-freq", "50000", "--resume",
    ]

    rollout = build_rollout_command(
        project_root=tmp_path,
        checkpoint=tmp_path / "checkpoints/100000/pretrained_model",
        dataset_root=tmp_path / "dataset",
        output_dir=tmp_path / "periodic_eval/step_100000",
        manifest=tmp_path / "periodic_eval/scenario_manifest.json",
        episodes=10,
        seed=42,
    )
    assert rollout[:4] == ["bash", str(tmp_path / "run.sh"), "rollout", "--headless"]
    assert ["--success-rate", "10"] == rollout[rollout.index("--success-rate"):rollout.index("--success-rate") + 2]
    assert "--scenario-manifest" in rollout
```

- [ ] **Step 2: Add a failing dry-run routing test**

Use `subprocess.run()` on `bash run.sh train-eval --dry-run --steps 180000` and assert stdout lists `50000, 100000, 150000, 180000`, contains no training/Isaac startup marker, and creates no `periodic_eval` files.

- [ ] **Step 3: Run tests and verify the interface is absent**

Run:

```bash
python3 -m pytest tests/test_periodic_eval.py -v
```

Expected: command helper imports or `train-eval` routing fail.

- [ ] **Step 4: Implement pure command builders**

`build_train_command()` returns an argv list and appends exactly one of `--resume` or `--no-resume`. `build_rollout_command()` must include:

```text
rollout --headless
--checkpoint <step checkpoint>/pretrained_model
--dataset-root <active converted dataset>
--success-rate 10
--seed 42
--scenario-manifest <shared manifest>
--output-dir <periodic_eval/step_NNNNNN>
--save-videos
--save-diagnostics
```

It may append current Rollout timing overrides, but defaults must remain the existing values.

- [ ] **Step 5: Implement the orchestration CLI**

`scripts/train_eval.py` must:

1. Load the active training config with `load_training_config()` and project config with `load_project_config()`.
2. Resolve total steps, save frequency, output directory, dataset directory, and node list.
3. Build a workflow fingerprint from task, dataset contract, training config, total steps, interval, evaluation count, seed, and Rollout timing parameters.
4. In `--dry-run`, print resolved paths and all nodes without creating directories or spawning children.
5. If a compatible workflow state exists, revalidate every completed node before skipping it.
6. If no state exists but a complete checkpoint exists, adopt only the latest checkpoint whose dataset contract matches; otherwise start fresh.
7. Refuse a non-empty incompatible output directory unless the user explicitly supplied `--overwrite-output`.
8. Run one child at a time with `subprocess.Popen(argv, stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)`, echo every line to stdout, and append it to the appropriate log.
9. Validate the exact checkpoint after training exits zero.
10. Run exactly one 10-episode Rollout process for the node, using the shared scenario manifest.
11. Validate artifacts, write `summary.csv`, and atomically mark the node complete.
12. Stop with a nonzero status on infrastructure failure; continue after valid unsuccessful task episodes.
13. Exit zero after the final target node is evaluated.

Do not use `shell=True`, do not send signals to training, and do not keep a Policy Server alive between nodes.

For a fresh run, do not create `OUTPUT_DIR/periodic_eval` before the first training child: the existing pure-training guard correctly rejects any non-empty fresh output directory. Stream the first segment into the sibling bootstrap log `OUTPUT_DIR.parent / f".{OUTPUT_DIR.name}.train_eval_bootstrap.log"`; after training creates a valid checkpoint, create `periodic_eval/step_050000/` and atomically move that log to `step_050000/train.log`. Subsequent resumed segments write directly to `step_NNNNNN/train.log`. Each Rollout writes to `step_NNNNNN/rollout.log`, and orchestration events append to `periodic_eval/train_eval.log`.

On every exception or nonzero child status, atomically store `status="failed"`, the active node, child kind, return code, error text, and timestamp in `workflow_state.json` when the output directory exists. On success store `status="complete"`, the latest checkpoint, manifest fingerprint, and the validated node list.

- [ ] **Step 6: Add `run.sh` routing and help**

Add to help:

```text
train-eval [options]              Train in segments and evaluate every checkpoint
```

Add routing that uses a lightweight Python interpreter and does not preselect the SmolVLA or IsaacLab GPU environment:

```bash
train-eval) shift; print_context; python3 scripts/train_eval.py "$@" ;;
```

Each spawned `bash run.sh train` or `bash run.sh rollout` command will select its own required environment.

- [ ] **Step 7: Run dry-run and unit tests**

Run:

```bash
python3 -m pytest tests/test_periodic_eval.py tests/test_rollout_metrics.py tests/test_policy_protocol.py -v
bash -n run.sh scripts/train_smolvla_local.sh
bash run.sh train-eval --dry-run --steps 180000
```

Expected: tests pass; dry-run prints four nodes and starts no training, Isaac Sim, or Policy Server process.

- [ ] **Step 8: Commit the orchestrator**

```bash
git add s4_smolvla_isaaclab/run.sh s4_smolvla_isaaclab/scripts/train_eval.py s4_smolvla_isaaclab/s4_pipeline/periodic_eval.py s4_smolvla_isaaclab/tests/test_periodic_eval.py
git commit -m "feat: add guarded periodic train-eval workflow"
```

---

### Task 6: Document periodic training, outputs, and recovery

**Files:**
- Modify: `docs/TRAINING.md`
- Modify: `docs/ONLINE_ROLLOUT.md`
- Modify: `docs/QUICKSTART.md`
- Modify: `docs/README.md`
- Modify: `docs/course/03_PROJECT_DEPLOYMENT.md`

**Interfaces:**
- Consumes: final `run.sh help`, `scripts/train_eval.py --help`, output layout, and recovery semantics.
- Produces: reproducible user instructions that distinguish pure training, periodic training/evaluation, task failure, and workflow failure.

- [ ] **Step 1: Capture the actual help output**

Run:

```bash
bash run.sh help
python3 scripts/train_eval.py --help
```

Expected: both commands exit zero without writing files or launching GPU processes.

- [ ] **Step 2: Document the normal command**

Use a verified command equivalent to:

```bash
cd /home/zfy/smolVLA/s4_smolvla_isaaclab
bash run.sh train-eval \
  --steps 500000 \
  --eval-interval 50000 \
  --eval-episodes 10 \
  --seed 42
```

Explain that headless Rollout still renders and saves camera videos, training and evaluation never run concurrently, and failed robot tasks count toward success rate without stopping the workflow.

- [ ] **Step 3: Document dry-run, resume, and failure handling**

Include:

```bash
bash run.sh train-eval --dry-run --steps 180000
bash run.sh train-eval --steps 500000 --eval-interval 50000 --eval-episodes 10 --seed 42
```

Explain automatic compatible-state recovery, when `--resume` is needed to adopt an existing training output, the destructive meaning of `--overwrite-output`, and which errors stop the workflow.

- [ ] **Step 4: Document artifacts and fair checkpoint comparison**

Document `scenario_manifest.json`, `workflow_state.json`, per-step logs, `.avi`, action CSV, action plot, `summary.json`, and `summary.csv`. State that every checkpoint uses the identical stored scenario samples, including randomized `[0.00, 0.05] m` drawer opening values.

- [ ] **Step 5: Verify all documented options exist**

Run:

```bash
rg -n "train-eval|eval-interval|eval-episodes|scenario_manifest|workflow_state" docs run.sh scripts/train_eval.py
bash run.sh train-eval --dry-run --steps 180000
git diff --check
```

Expected: documented names match actual CLI options; dry-run succeeds; no Markdown whitespace errors.

- [ ] **Step 6: Commit documentation**

```bash
git add s4_smolvla_isaaclab/docs
git commit -m "docs: explain periodic train-eval workflow"
```

---

### Task 7: Run the final non-GPU verification gate

**Files:**
- Modify only if verification proves a narrowly scoped defect.

**Interfaces:**
- Consumes: completed periodic workflow implementation.
- Produces: test evidence without real training, Isaac Sim, or checkpoint loading.

- [ ] **Step 1: Run the complete pure test suite**

Run:

```bash
python3 -m pytest tests -v
```

Expected: all pure tests pass; environment-dependent skips are reported explicitly.

- [ ] **Step 2: Run static and dry-run checks**

Run:

```bash
bash -n run.sh scripts/*.sh
bash run.sh help
bash run.sh train-eval --dry-run --steps 180000 --eval-interval 50000 --eval-episodes 10 --seed 42
git diff --check
```

Expected: Shell parsing succeeds; dry-run lists exactly 50k, 100k, 150k, and 180k; it creates no workflow directory and starts no GPU process.

- [ ] **Step 3: Exercise failure fixtures only**

Run the exact unit tests for incomplete checkpoint, missing rollout artifact, scenario fingerprint mismatch, and incompatible workflow state:

```bash
python3 -m pytest tests/test_periodic_eval.py -v -k "checkpoint or artifact or manifest or fingerprint"
```

Expected: all tests pass by proving the validators reject their invalid fixtures.

- [ ] **Step 4: Verify external repositories remain unchanged by this feature**

Run:

```bash
git -C /home/zfy/smolVLA/lerobot status --short
git -C /home/zfy/IsaacLab status --short
```

Expected: no change introduced by this implementation. Report pre-existing unrelated changes without touching them.

- [ ] **Step 5: Provide the real user-run smoke command without executing it**

Hand off:

```bash
cd /home/zfy/smolVLA/s4_smolvla_isaaclab
bash run.sh train-eval \
  --steps 500000 \
  --eval-interval 50000 \
  --eval-episodes 10 \
  --seed 42
```

State clearly that this is the first real GPU/integration verification and was not run during implementation.

- [ ] **Step 6: Commit any final narrow correction**

If verification required a correction, stage only the corrected files and commit:

```bash
git commit -m "test: harden periodic train-eval verification"
```

If no correction was required, do not create an empty commit.
