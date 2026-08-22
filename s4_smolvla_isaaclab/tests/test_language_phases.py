from __future__ import annotations

import importlib

import pytest

from tasks.drawer_insert_close_controller import load_scripted_config


EXPECTED_IDS = (
    "prepare_hands",
    "approach_drawer_handle",
    "grasp_drawer_handle",
    "pull_drawer",
    "approach_can",
    "grasp_can",
    "lift_can",
    "place_can",
    "release_and_retreat",
    "close_drawer_and_home",
)


def _language_module():
    return importlib.import_module("s4_pipeline.language_phases")


def test_drawer_language_contract_covers_all_expert_phases_once_in_order():
    language = _language_module()
    scripted = load_scripted_config()
    contract = language.load_language_phase_contract(scripted)

    assert contract.version == "drawer_10phase_v1"
    assert tuple(phase.id for phase in contract.phases) == EXPECTED_IDS
    assert len({phase.task for phase in contract.phases}) == 10

    expert_names = tuple(item["name"] for item in scripted["phases"])
    mapped_names = tuple(name for phase in contract.phases for name in phase.source_phases)
    assert mapped_names == expert_names


def test_drawer_language_contract_maps_expert_name_legacy_text_and_prompt():
    language = _language_module()
    scripted = load_scripted_config()
    contract = language.load_language_phase_contract(scripted)
    expert_by_name = {item["name"]: item for item in scripted["phases"]}

    assert contract.for_expert_phase("left_approach_handle").id == "approach_drawer_handle"
    assert contract.for_expert_phase("left_grasp_handle").id == "approach_drawer_handle"
    legacy = expert_by_name["right_settle_before_close"]["task"]
    assert contract.for_legacy_task(legacy).id == "approach_can"
    prompt = contract.for_id("release_and_retreat").task
    assert contract.for_prompt(prompt).id == "release_and_retreat"
    assert contract.rollout_gate_config("close_drawer_and_home", scripted)["name"] == "left_home"


def test_language_contract_rejects_duplicate_or_incomplete_source_coverage():
    language = _language_module()
    scripted = load_scripted_config()
    broken = dict(scripted)
    broken["language_phases"] = [dict(item) for item in scripted["language_phases"]]
    broken["language_phases"][1] = dict(broken["language_phases"][1])
    broken["language_phases"][1]["source_phases"] = [
        "initial_open_hands",
        *broken["language_phases"][1]["source_phases"],
    ]

    with pytest.raises(ValueError, match="exactly once"):
        language.load_language_phase_contract(broken)


def test_language_contract_rejects_unknown_lookup_values():
    language = _language_module()
    contract = language.load_language_phase_contract(load_scripted_config())

    with pytest.raises(ValueError, match="Unknown expert phase"):
        contract.for_expert_phase("missing")
    with pytest.raises(ValueError, match="Unknown recorded task"):
        contract.for_legacy_task("missing task")
    with pytest.raises(ValueError, match="Unknown language phase"):
        contract.for_id("missing")
