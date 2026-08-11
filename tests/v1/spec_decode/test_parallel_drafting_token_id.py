# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


@pytest.mark.parametrize(
    ("hf_config", "expected"),
    [
        (SimpleNamespace(mask_token_id=11), 11),
        (SimpleNamespace(dspark_noise_token_id=22), 22),
        (SimpleNamespace(pard_token=33), 33),
        (SimpleNamespace(ptd_token_id=44), 44),
        (SimpleNamespace(dflash_config={"mask_token_id": 55}), 55),
    ],
)
def test_v1_parallel_drafting_token_id_resolution(hf_config, expected):
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.draft_model_config = SimpleNamespace(hf_config=hf_config)
    proposer.pass_hidden_states_to_model = False

    proposer._init_parallel_drafting_params()

    assert proposer.parallel_drafting_token_id == expected


def test_v1_parallel_drafting_token_id_missing_fails_closed():
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.draft_model_config = SimpleNamespace(hf_config=SimpleNamespace())
    proposer.pass_hidden_states_to_model = False

    with pytest.raises(ValueError, match="dspark_noise_token_id"):
        proposer._init_parallel_drafting_params()
