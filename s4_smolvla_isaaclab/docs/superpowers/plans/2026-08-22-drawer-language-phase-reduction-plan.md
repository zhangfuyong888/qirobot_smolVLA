# Drawer 10-Phase Language Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the 20-stage expert controller while converting, training, and rolling out with a stable 10-stage language contract.

**Architecture:** A project-owned pure helper parses the ordered language contract from the scripted YAML and exposes mappings by stable phase ID, prompt, legacy task text, and expert phase name. Collection records macro prompt plus stable/expert IDs; conversion relabels both legacy and new HDF5 into a new dataset; Policy Server and Rollout use stable IDs instead of prompt equality.

**Tech Stack:** Python 3.11/3.12, YAML, HDF5/h5py, PyArrow, LeRobotDataset API, pytest, Bash.

**Spec:** `docs/superpowers/specs/2026-08-22-drawer-language-phase-reduction-design.md`

## Global Constraints

- Do not modify `/home/zfy/smolVLA/lerobot`.
- Keep all 20 expert control phases, trajectories, gates, randomization, and success criteria unchanged.
- Do not overwrite the existing HDF5, `s4_drawer_insert_close_v0`, old outputs, or checkpoints.
- New dataset ID: `s4_drawer_insert_close_v1_10phase`.
- New training output: `smolvla_drawer_insert_close_v1_10phase`.
- Keep state/action 26D, absolute joint targets, three cameras, 20 Hz data, 120 Hz control, and 50-frame chunks.
- Use stable language phase IDs as program keys; prompts are model inputs only.
- Preserve legacy HDF5 conversion by mapping its per-frame old task text.
- Do not execute the 103 GB conversion, training, Isaac Sim, or Rollout during code verification.

---

### Task 1: Language contract parser and validated 10-stage configuration

**Files:**
- Create: `s4_pipeline/language_phases.py`
- Modify: `configs/tasks/drawer_insert_close.scripted.yaml`
- Modify: `tasks/drawer_insert_close_controller.py`
- Test: `tests/test_language_phases.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `LanguagePhase`, `LanguagePhaseContract`, `load_language_phase_contract(scripted_cfg)`.
- Produces mappings: `for_expert_phase(name)`, `for_legacy_task(text)`, `for_prompt(text)`, and `rollout_gate_config(phase_id, scripted_cfg)`.
- Controller exposes `current_language_phase_id` and `current_language_task` without changing its existing `step()` return tuple.

- [ ] Write failing tests with a 20-stage fixture proving exactly 10 ordered phases, unique IDs/prompts, complete one-time source coverage, contiguous source groups, legacy text lookup, and gate-source membership.
- [ ] Run `python3 -m pytest tests/test_language_phases.py tests/test_config.py -q` and confirm failure because the parser/config does not exist.
- [ ] Add the 10 approved language phases to YAML, including `id`, `task`, `source_phases`, and `rollout_gate_phase`.
- [ ] Implement immutable contract objects and validation. Unknown phase IDs/texts must raise `ValueError`; no silent fallback.
- [ ] Load the contract in `DrawerInsertCloseController` and expose the current macro ID/prompt based on `current_phase.name`.
- [ ] Re-run the focused tests and confirm they pass.
- [ ] Commit only project files with `git commit -m "feat: define drawer language phase contract"` if repository metadata is writable.

### Task 2: Record stable language and expert phase fields in new HDF5

**Files:**
- Modify: `data/hdf5_schema.py`
- Modify: `data/dataset_writer.py`
- Modify: `scripts/record_dataset.py`
- Test: `tests/test_hdf5_schema.py`
- Test: `tests/test_drawer_grasp_contract.py`

**Interfaces:**
- Adds schema paths `obs/language_phase_id` and `obs/expert_phase_name`.
- `EpisodeBuffer` adds `language_phase_ids: list[str]` and `expert_phase_names: list[str]`.
- `append_bimanual_record_frame(...)` accepts `language_phase_id` and `expert_phase_name` in addition to the macro task prompt.

- [ ] Write failing HDF5 tests that create a small episode and assert all three equal-length string arrays are persisted.
- [ ] Run the focused tests and confirm missing fields fail.
- [ ] Add constants, buffer fields, length validation, and UTF-8 HDF5 writes.
- [ ] Change the recorder to pass `drawer_controller.current_language_phase_id`, `current_language_task`, and the actual `current_scripted_phase` for every recorded frame.
- [ ] Change diagnostics that infer expert phase from task text to read the parallel expert-phase buffer, with legacy fallback only when needed.
- [ ] Add the language contract version and ordered IDs to collection `env_args` so resume rejects mixed 20/10-language HDF5.
- [ ] Run the focused HDF5/contract tests and confirm they pass.
- [ ] Commit with `git commit -m "feat: record stable drawer language phases"` if Git metadata is writable.

### Task 3: Conversion-time legacy relabeling and portable dataset contract

**Files:**
- Modify: `data/lerobot_conversion.py`
- Modify: `scripts/convert_lerobot.py`
- Modify: `configs/tasks/drawer_insert_close.dataset.json`
- Test: `tests/test_video.py`
- Create or modify: `tests/test_lerobot_conversion.py`

**Interfaces:**
- `convert_hdf5_to_lerobot(..., language_contract: LanguagePhaseContract | None)` maps each frame before calling `dataset.add_frame`.
- Mapping priority: HDF5 language ID, then legacy task text; unknown values raise with file/demo/frame context.
- `meta/s4_contract.json` adds `language_contract_version` and ordered `language_phases` records.

- [ ] Write a failing lightweight conversion/mapping test for legacy task strings, new phase IDs, and unknown task rejection.
- [ ] Run the focused conversion tests and verify the expected failure.
- [ ] Load the scripted configuration in `convert_lerobot.py`, build the language contract, and pass it into conversion.
- [ ] Implement per-frame mapping without modifying state, action, timestamps, or images.
- [ ] Write the portable ordered language contract sidecar.
- [ ] Change active dataset/staging/root IDs to `s4_drawer_insert_close_v1_10phase`, leaving legacy files untouched and documenting use of `--root-path` for one-time conversion.
- [ ] Verify focused tests pass and that `git diff` contains no `lerobot/` paths.
- [ ] Commit with `git commit -m "feat: relabel drawer language during conversion"` if possible.

### Task 4: Stable-ID Policy Server schedule and Rollout gate resolution

**Files:**
- Modify: `scripts/policy_server.py`
- Modify: `scripts/eval_policy.py`
- Modify: `s4_pipeline/rollout_metrics.py`
- Test: `tests/test_policy_protocol.py`
- Test: `tests/test_rollout_metrics.py`
- Test: `tests/test_language_phases.py`

**Interfaces:**
- `_load_phase_schedule()` returns each item with `language_phase_id`, `task`, `task_index`, `phase_index`, and median `frames`.
- Legacy datasets without a language contract retain prompt-based schedules.
- Rollout resolves `rollout_gate_phase` from stable ID and uses the source expert phase configuration.
- `rollout_phase_extension_frames()` keys 80-frame overrides by stable ID, with legacy fallback.

- [ ] Write failing temporary-Parquet tests showing adjacent source labels collapse into 10 ordered macro runs and stable IDs are returned.
- [ ] Write failing gate-resolution tests proving macro phases use their terminal expert phase and unknown IDs fail.
- [ ] Extend the existing 80-frame test to use `approach_drawer_handle` and `pull_drawer` IDs while unrelated phases remain 20.
- [ ] Run focused tests and confirm failures identify missing stable-ID behavior.
- [ ] Implement contract loading in Policy Server and include phase IDs in the ready message schedule.
- [ ] Refactor Rollout gate lookup and total extension budget to stable IDs; retain explicit legacy behavior for old dataset/checkpoint pairs.
- [ ] Confirm stage switching still clears chunks, resets policy, predicts 50 actions, blends 8 frames, and replans every 40 frames.
- [ ] Run focused protocol/rollout tests and confirm they pass.
- [ ] Commit with `git commit -m "feat: drive drawer rollout by language phase ids"` if possible.

### Task 5: Dataset validation and mixed-contract rejection

**Files:**
- Modify: `scripts/dataset_check.py`
- Modify: `s4_pipeline/language_phases.py`
- Test: `tests/test_contract.py`
- Create or modify: `tests/test_dataset_check.py`

**Interfaces:**
- Pure validation consumes ordered per-episode task runs plus `meta/s4_contract.json` and returns no value or raises a precise `ValueError`.
- New datasets must have the exact 10 prompts and order; legacy datasets are reported as legacy rather than silently accepted as new.

- [ ] Write failing tests for correct 10-stage runs, unknown prompt, mixed old/new prompts, missing contract, duplicate/out-of-order run, and incomplete phase coverage.
- [ ] Run tests and verify expected failures.
- [ ] Implement validation and call it from the LeRobotDataset checker after loading task metadata and frame rows.
- [ ] Include dataset path, episode index, expected ID, and observed prompt in errors.
- [ ] Preserve existing 26D/camera/FPS/checkpoint processor checks.
- [ ] Run focused tests and confirm pass.
- [ ] Commit with `git commit -m "feat: validate drawer language dataset contract"` if possible.

### Task 6: New training output and checkpoint language compatibility metadata

**Files:**
- Modify: `configs/tasks/drawer_insert_close.smolvla.yaml`
- Modify: `scripts/train_smolvla_local.sh`
- Modify: `scripts/dataset_check.py`
- Test: `tests/test_config.py`
- Test: `tests/test_contract.py`

**Interfaces:**
- Training reads `s4_drawer_insert_close_v1_10phase` and writes `smolvla_drawer_insert_close_v1_10phase`.
- Training launch records/copies the dataset language contract into the run directory without modifying LeRobot trainer code.
- Dataset/checkpoint validation compares project-owned sidecars when present.

- [ ] Write failing config tests for new dataset and output IDs and for rejecting resume from legacy-language output.
- [ ] Run focused tests and confirm failure against v0 configuration.
- [ ] Update training YAML and add a preflight that requires the 10-stage dataset contract.
- [ ] Copy the contract to the training output as a project-owned provenance sidecar before/after invoking the existing trainer; never patch LeRobot.
- [ ] Add compatibility checks that do not reject historical checkpoints lacking the sidecar unless paired with the new dataset.
- [ ] Run focused tests and Shell syntax checks.
- [ ] Commit with `git commit -m "feat: version ten-phase drawer training"` if possible.

### Task 7: User documentation and migration commands

**Files:**
- Modify: `README.md`
- Modify: `docs/DATA_COLLECTION.md`
- Modify: `docs/DATA_SCHEMA.md`
- Modify: `docs/DATASET_CONVERSION.md`
- Modify: `docs/DATASET_VALIDATION.md`
- Modify: `docs/TRAINING.md`
- Modify: `docs/ONLINE_ROLLOUT.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/knowledge_base/CORE_CONTRACTS.md`
- Modify: `docs/knowledge_base/END_TO_END_PIPELINE.md`
- Modify: `docs/knowledge_base/POLICY_SERVER_AND_ROLLOUT.md`
- Modify: `docs/course/02_PROJECT_IMPLEMENTATION.md`

**Interfaces:**
- Documents distinguish 20 expert control stages from 10 language stages.
- Provides commands for legacy HDF5 conversion, future collection, validation, fresh training, and Rollout.

- [ ] Update the central architecture/data-contract descriptions and add the exact 10-stage mapping table once.
- [ ] Link detailed pages to the central table rather than duplicating conflicting lists.
- [ ] Add a one-time conversion command using the existing legacy HDF5 path and the new repo ID; mark `--overwrite` destructive only to the new target.
- [ ] State that fresh training is required and old checkpoint resume is unsupported for the new language contract.
- [ ] State that direct Rollout requires the matching new dataset plus checkpoint.
- [ ] Run Markdown link/path searches and verify no active instructions claim 20 training prompts for the new dataset.
- [ ] Commit with `git commit -m "docs: document ten-phase drawer pipeline"` if possible.

### Task 8: Full verification without heavy workloads

**Files:**
- Verify all modified project files.
- Do not modify or test files under `lerobot/`.

**Interfaces:**
- Produces evidence for CPU tests, syntax, contracts, and repository scope.

- [ ] Run all project CPU tests with the available project Python and report exact pass/fail counts.
- [ ] Run `bash -n run.sh scripts/*.sh environment/*.sh`.
- [ ] Parse all modified Python files with `ast.parse` or run the configured linter if available.
- [ ] Run `git diff --check`.
- [ ] Run `git status --short` and `git -C lerobot status --short`; confirm no LeRobot modification.
- [ ] Validate the active 10-stage config and print the resolved mapping/order.
- [ ] Perform a tiny synthetic HDF5→LeRobot conversion if the test fixture supports it; do not read/re-encode the 103 GB real HDF5.
- [ ] Do not claim Isaac Sim, full conversion, training, or Rollout success because those heavy workflows are intentionally not executed.
- [ ] Report the exact command the user should run for real legacy conversion, dataset check, fresh training, and Rollout.
