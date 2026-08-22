import base64

import numpy as np

from scripts.policy_server import _annotate_phase_schedule, _image_array_from_payload
from s4_pipeline.language_phases import load_language_phase_contract
from tasks.drawer_insert_close_controller import load_scripted_config


def test_json_image_payload_roundtrip():
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    payload = {"shape": list(image.shape), "b64": base64.b64encode(image.tobytes()).decode("ascii")}
    decoded = _image_array_from_payload(payload)
    assert np.array_equal(decoded, image)
    assert decoded.flags.writeable


def test_phase_schedule_is_annotated_with_stable_macro_ids():
    contract = load_language_phase_contract(load_scripted_config())
    schedule = [
        {"phase_index": index, "task_index": index, "task": phase.task, "frames": 10}
        for index, phase in enumerate(contract.phases)
    ]
    annotated = _annotate_phase_schedule(
        schedule,
        {
            "language_contract_version": contract.version,
            "language_phases": contract.as_portable_records(),
        },
    )
    assert [item["language_phase_id"] for item in annotated] == [
        phase.id for phase in contract.phases
    ]


def test_phase_schedule_rejects_prompt_not_in_dataset_contract():
    contract = load_language_phase_contract(load_scripted_config())
    schedule = [{"phase_index": 0, "task_index": 0, "task": "unknown", "frames": 10}]
    import pytest

    with pytest.raises(ValueError, match="not present in language contract"):
        _annotate_phase_schedule(
            schedule,
            {
                "language_contract_version": contract.version,
                "language_phases": contract.as_portable_records(),
            },
        )
