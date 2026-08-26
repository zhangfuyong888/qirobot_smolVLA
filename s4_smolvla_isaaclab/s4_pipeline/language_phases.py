"""Validated language-stage contracts shared by collection, conversion, and rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ACTION_GROUPS = frozenset({"left_arm", "left_hand", "right_arm", "right_hand"})
ROLLOUT_TIMEOUTS = frozenset({"advance", "fail"})
ROLLOUT_EXTENSIONS = frozenset({"default", "drawer"})
ROLLOUT_FAILURE_CONDITIONS = frozenset({"none", "drawer_open_min"})


@dataclass(frozen=True)
class LanguagePhase:
    id: str
    task: str
    source_phases: tuple[str, ...]
    rollout_gate_phase: str
    active_action_groups: tuple[str, ...]
    rollout_timeout: str
    rollout_extension: str
    rollout_failure_condition: str


@dataclass(frozen=True)
class LanguagePhaseContract:
    version: str
    phases: tuple[LanguagePhase, ...]
    _by_id: dict[str, LanguagePhase]
    _by_prompt: dict[str, LanguagePhase]
    _by_expert_phase: dict[str, LanguagePhase]
    _by_legacy_task: dict[str, LanguagePhase]

    def for_id(self, phase_id: str) -> LanguagePhase:
        try:
            return self._by_id[str(phase_id)]
        except KeyError as exc:
            raise ValueError(f"Unknown language phase ID: {phase_id!r}") from exc

    def for_prompt(self, prompt: str) -> LanguagePhase:
        try:
            return self._by_prompt[str(prompt)]
        except KeyError as exc:
            raise ValueError(f"Unknown language prompt: {prompt!r}") from exc

    def for_expert_phase(self, phase_name: str) -> LanguagePhase:
        try:
            return self._by_expert_phase[str(phase_name)]
        except KeyError as exc:
            raise ValueError(f"Unknown expert phase: {phase_name!r}") from exc

    def for_legacy_task(self, task: str) -> LanguagePhase:
        try:
            return self._by_legacy_task[str(task)]
        except KeyError as exc:
            raise ValueError(f"Unknown recorded task: {task!r}") from exc

    def resolve_recorded_task(self, task: str) -> LanguagePhase:
        """Resolve either a new macro prompt or a legacy expert-stage prompt."""
        value = str(task)
        if value in self._by_prompt:
            return self._by_prompt[value]
        return self.for_legacy_task(value)

    def rollout_gate_config(self, phase_id: str, scripted_cfg: dict[str, Any]) -> dict[str, Any]:
        phase = self.for_id(phase_id)
        expert_by_name = {
            str(item.get("name", "")): item for item in scripted_cfg.get("phases", [])
        }
        try:
            return expert_by_name[phase.rollout_gate_phase]
        except KeyError as exc:
            raise ValueError(
                f"Language phase {phase.id!r} references unknown rollout gate phase "
                f"{phase.rollout_gate_phase!r}"
            ) from exc

    def as_portable_records(self) -> list[dict[str, Any]]:
        return [
            {
                "id": phase.id,
                "task": phase.task,
                "source_phases": list(phase.source_phases),
                "rollout_gate_phase": phase.rollout_gate_phase,
                "active_action_groups": list(phase.active_action_groups),
                "rollout_timeout": phase.rollout_timeout,
                "rollout_extension": phase.rollout_extension,
                "rollout_failure_condition": phase.rollout_failure_condition,
            }
            for phase in self.phases
        ]


def load_language_phase_contract(scripted_cfg: dict[str, Any]) -> LanguagePhaseContract:
    """Parse and validate the ordered language contract in a scripted task config."""
    version = str(scripted_cfg.get("language_contract_version", "")).strip()
    if not version:
        raise ValueError("language_contract_version must be a non-empty string")
    raw_language_phases = scripted_cfg.get("language_phases")
    if not isinstance(raw_language_phases, list) or not raw_language_phases:
        raise ValueError("language_phases must be a non-empty ordered list")
    raw_expert_phases = scripted_cfg.get("phases")
    if not isinstance(raw_expert_phases, list) or not raw_expert_phases:
        raise ValueError("phases must be a non-empty ordered list")

    expert_names = tuple(str(item.get("name", "")) for item in raw_expert_phases)
    if any(not name for name in expert_names) or len(set(expert_names)) != len(expert_names):
        raise ValueError("Expert phase names must be non-empty and unique")
    expert_by_name = {str(item["name"]): item for item in raw_expert_phases}
    expert_tasks = {str(item["name"]): str(item.get("task", "")) for item in raw_expert_phases}
    phases: list[LanguagePhase] = []
    for raw in raw_language_phases:
        if not isinstance(raw, dict):
            raise ValueError("Each language phase must be a mapping")
        phase_id = str(raw.get("id", "")).strip()
        task = str(raw.get("task", "")).strip()
        sources_raw = raw.get("source_phases")
        if not phase_id or not task:
            raise ValueError("Language phase id and task must be non-empty")
        if not isinstance(sources_raw, list) or not sources_raw:
            raise ValueError(f"Language phase {phase_id!r} requires source_phases")
        source_phases = tuple(str(name) for name in sources_raw)
        rollout_gate_phase = str(raw.get("rollout_gate_phase", "")).strip()
        if rollout_gate_phase not in source_phases:
            raise ValueError(
                f"Language phase {phase_id!r} rollout_gate_phase must belong to source_phases"
            )
        active_raw = raw.get("active_action_groups")
        if not isinstance(active_raw, list) or not active_raw:
            raise ValueError(f"Language phase {phase_id!r} requires active_action_groups")
        active_action_groups = tuple(str(group) for group in active_raw)
        if len(set(active_action_groups)) != len(active_action_groups):
            raise ValueError(f"Language phase {phase_id!r} has duplicate active_action_groups")
        unknown_groups = sorted(set(active_action_groups) - ACTION_GROUPS)
        if unknown_groups:
            raise ValueError(
                f"Language phase {phase_id!r} has unknown active_action_groups={unknown_groups}"
            )
        rollout_timeout = str(raw.get("rollout_timeout", "fail")).strip()
        if rollout_timeout not in ROLLOUT_TIMEOUTS:
            raise ValueError(
                f"Language phase {phase_id!r} rollout_timeout must be one of "
                f"{sorted(ROLLOUT_TIMEOUTS)}"
            )
        rollout_extension = str(raw.get("rollout_extension", "default")).strip()
        if rollout_extension not in ROLLOUT_EXTENSIONS:
            raise ValueError(
                f"Language phase {phase_id!r} rollout_extension must be one of "
                f"{sorted(ROLLOUT_EXTENSIONS)}"
            )
        rollout_failure_condition = str(
            raw.get("rollout_failure_condition", "none")
        ).strip()
        if rollout_failure_condition not in ROLLOUT_FAILURE_CONDITIONS:
            raise ValueError(
                f"Language phase {phase_id!r} rollout_failure_condition must be one of "
                f"{sorted(ROLLOUT_FAILURE_CONDITIONS)}"
            )
        if rollout_failure_condition == "drawer_open_min":
            gate_cfg = expert_by_name[rollout_gate_phase]
            if gate_cfg.get("drawer_open_min") is None:
                raise ValueError(
                    f"Language phase {phase_id!r} uses drawer_open_min failure condition "
                    f"but gate phase {rollout_gate_phase!r} has no drawer_open_min"
                )
        phases.append(
            LanguagePhase(
                phase_id,
                task,
                source_phases,
                rollout_gate_phase,
                active_action_groups,
                rollout_timeout,
                rollout_extension,
                rollout_failure_condition,
            )
        )

    ids = [phase.id for phase in phases]
    prompts = [phase.task for phase in phases]
    if len(set(ids)) != len(ids):
        raise ValueError("Language phase IDs must be unique")
    if len(set(prompts)) != len(prompts):
        raise ValueError("Language phase tasks must be unique")
    mapped_expert_names = tuple(name for phase in phases for name in phase.source_phases)
    if mapped_expert_names != expert_names:
        raise ValueError(
            "Expert phases must appear exactly once in language_phases and preserve configured order"
        )

    by_id = {phase.id: phase for phase in phases}
    by_prompt = {phase.task: phase for phase in phases}
    by_expert_phase = {
        expert_name: phase for phase in phases for expert_name in phase.source_phases
    }
    by_legacy_task: dict[str, LanguagePhase] = {}
    for expert_name, task in expert_tasks.items():
        if not task:
            raise ValueError(f"Expert phase {expert_name!r} requires a legacy task string")
        if task in by_legacy_task:
            raise ValueError(f"Legacy expert task strings must be unique: {task!r}")
        by_legacy_task[task] = by_expert_phase[expert_name]
    return LanguagePhaseContract(
        version=version,
        phases=tuple(phases),
        _by_id=by_id,
        _by_prompt=by_prompt,
        _by_expert_phase=by_expert_phase,
        _by_legacy_task=by_legacy_task,
    )
