# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
from transformers import AutoTokenizer

from vllm.config import DeviceConfig, StructuredOutputsConfig, VllmConfig
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_xgrammar import (
    XgrammarBackend,
    XgrammarGrammar,
)

TOKENIZER = "openai-community/gpt2"
VOCAB_SIZE = 50257
EOS = 50256
QUOTE = 1
LETTER = 55


def _token_allowed(row, token_id: int) -> bool:
    word = int(row[token_id // 32].item()) & 0xFFFFFFFF
    return bool(word & (1 << (token_id % 32)))


@pytest.fixture(scope="module")
def backend() -> XgrammarBackend:
    vllm_config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        structured_outputs_config=StructuredOutputsConfig(backend="xgrammar"),
    )
    return XgrammarBackend(
        vllm_config,
        tokenizer=AutoTokenizer.from_pretrained(TOKENIZER),
        vocab_size=VOCAB_SIZE,
    )


def test_request_stop_tokens_gated_to_grammar_terminal(backend: XgrammarBackend):
    schema = '{"type": "string"}'
    default = backend.compile_grammar(StructuredOutputOptions.JSON, schema)
    override = backend.compile_grammar(
        StructuredOutputOptions.JSON,
        schema,
        stop_token_ids={EOS, LETTER},
    )

    for grammar in (default, override):
        assert grammar.accept_tokens("req", [QUOTE])

    bm_default = backend.allocate_token_bitmask(1)
    bm_override = backend.allocate_token_bitmask(1)
    default.fill_bitmask(bm_default, 0)
    override.fill_bitmask(bm_override, 0)

    assert _token_allowed(bm_default[0], LETTER)
    assert not _token_allowed(bm_override[0], LETTER)

    for grammar in (default, override):
        assert grammar.accept_tokens("req", [QUOTE])
        assert not grammar.is_terminated()

    default.fill_bitmask(bm_default, 0)
    override.fill_bitmask(bm_override, 0)

    assert not _token_allowed(bm_default[0], LETTER)
    assert _token_allowed(bm_override[0], LETTER)
    assert _token_allowed(bm_default[0], EOS)
    assert _token_allowed(bm_override[0], EOS)


def test_rejected_token_synchronizes_terminated_state():
    """A rejected token cannot leave the cached matcher state stale."""
    matcher = Mock()
    matcher.is_terminated.side_effect = [False, True]
    matcher.accept_token.return_value = False
    grammar = XgrammarGrammar(vocab_size=1, matcher=matcher, ctx=Mock())

    assert not grammar.accept_tokens("req", [7])
    assert grammar.is_terminated()
