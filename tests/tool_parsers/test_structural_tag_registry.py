# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai.types.responses import FunctionTool
from xgrammar import Grammar, StructuralTag
from xgrammar.structural_tag import TagsWithSeparatorFormat, TriggeredTagsFormat

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.abstract_parser import DelegatingParser
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.deepseekv3_tool_parser import DeepSeekV3ToolParser
from vllm.tool_parsers.deepseekv4_tool_parser import DeepSeekV4ToolParser
from vllm.tool_parsers.deepseekv31_tool_parser import DeepSeekV31ToolParser
from vllm.tool_parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
from vllm.tool_parsers.kimi_k2_tool_parser import KimiK2ToolParser
from vllm.tool_parsers.llama_tool_parser import Llama3JsonToolParser
from vllm.tool_parsers.minimax_m2_tool_parser import MinimaxM2ToolParser
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser
from vllm.tool_parsers.structural_tag_registry import (
    SUPPORTED_STRUCTURAL_TAG_MODELS,
    VLLM_BUILTIN_STRUCTURAL_TAG_MODELS,
    XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS,
    _get_function_parameters,
    get_model_structural_tag,
)


@pytest.fixture
def sample_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        )
    ]


@pytest.fixture
def sample_tools_strict() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        )
    ]


def test_supported_structural_tag_models_include_vllm_builtins():
    assert SUPPORTED_STRUCTURAL_TAG_MODELS == (
        XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS | VLLM_BUILTIN_STRUCTURAL_TAG_MODELS
    )
    assert "hermes" in VLLM_BUILTIN_STRUCTURAL_TAG_MODELS


@pytest.mark.parametrize("model", sorted(XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS))
def test_get_model_structural_tag_supports_all_xgrammar_builtins(
    model: str,
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model=model,
        tools=sample_tools_strict,
        tool_choice="auto",
        reasoning=False,
    )

    assert isinstance(tag, StructuralTag)


def test_get_model_structural_tag_supports_vllm_hermes(
    sample_tools: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model="hermes",
        tools=sample_tools,
        tool_choice="required",
        reasoning=False,
    )

    assert isinstance(tag, StructuralTag)
    assert tag.model_dump() == {
        "type": "structural_tag",
        "format": {
            "type": "tags_with_separator",
            "tags": [
                {
                    "type": "tag",
                    "begin": '<tool_call>\n{"name": "get_weather", "arguments": ',
                    "content": {
                        "type": "json_schema",
                        "json_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                            },
                            "style": "json",
                            "any_order": False,
                        },
                    "end": "}\n</tool_call>",
                },
                {
                    "type": "tag",
                    "begin": '<tool_call>{"name": "get_weather", "arguments": ',
                    "content": {
                        "type": "json_schema",
                        "json_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                            },
                            "style": "json",
                            "any_order": False,
                        },
                    "end": "}</tool_call>",
                },
            ],
            "separator": "",
            "at_least_one": True,
            "stop_after_first": False,
        },
    }


def test_hermes_required_tool_calls_use_empty_separator():
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_time",
                "parameters": {"type": "object", "properties": {}},
            },
        ),
    ]

    tag = get_model_structural_tag(
        model="hermes",
        tools=tools,
        tool_choice="required",
        reasoning=False,
    )

    assert tag is not None
    assert tag.format.separator == ""


@pytest.mark.parametrize("model", sorted(XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS))
def test_get_model_structural_tag_supports_named_tool_choice(
    model: str,
    sample_tools: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model=model,
        tools=sample_tools,
        tool_choice=ChatCompletionNamedToolChoiceParam(
            function=ChatCompletionNamedFunction(name="get_weather")
        ),
        reasoning=False,
    )

    assert isinstance(tag, StructuralTag)


@pytest.mark.parametrize(
    ("parser_cls", "model"),
    [
        (DeepSeekV3ToolParser, "deepseek_r1"),
        (DeepSeekV31ToolParser, "deepseek_v3_1"),
        (DeepSeekV32ToolParser, "deepseek_v3_2"),
        (DeepSeekV4ToolParser, "deepseek_v4"),
        (Glm47MoeModelToolParser, "glm_4_7"),
        (Hermes2ProToolParser, "hermes"),
        (KimiK2ToolParser, "kimi"),
        (Llama3JsonToolParser, "llama"),
        (MinimaxM2ToolParser, "minimax"),
        (Qwen3EngineToolParser, "qwen_3_coder"),
    ],
)
def test_tool_parsers_declare_matching_xgrammar_builtin_model(parser_cls, model):
    assert parser_cls.structural_tag_model == model
    assert not parser_cls.supports_required_and_named


def test_tool_parsers_without_structural_tag_support_required_and_named():
    class NonStructuralTagToolParser(ToolParser):
        pass

    assert NonStructuralTagToolParser.structural_tag_model is None
    assert NonStructuralTagToolParser.supports_required_and_named


def test_non_structural_tag_parser_uses_schema_constraints(
    sample_tools: list[ChatCompletionToolsParam],
):
    parser = ToolParser(MagicMock())
    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        tool_choice="required",
    )

    out = parser.adjust_request(request)

    assert out.structured_outputs is not None
    assert out.structured_outputs.json is not None
    assert out.structured_outputs.structural_tag is None


def test_get_structural_tag_disables_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    captured: list[bool] = []

    def fake_get_model_structural_tag(*, reasoning: bool, **kwargs):
        captured.append(reasoning)
        return None

    monkeypatch.setattr(
        "vllm.tool_parsers.structural_tag_registry.get_model_structural_tag",
        fake_get_model_structural_tag,
    )

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools_strict,
        tool_choice="auto",
    )
    parser = Qwen3EngineToolParser(MagicMock(), tools=sample_tools_strict)

    parser.get_structural_tag(request)

    assert captured == [False]


def test_unified_parser_get_structural_tag_disables_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    captured: list[bool] = []

    def fake_get_model_structural_tag(*, reasoning: bool, **kwargs):
        captured.append(reasoning)
        return None

    monkeypatch.setattr(
        "vllm.tool_parsers.structural_tag_registry.get_model_structural_tag",
        fake_get_model_structural_tag,
    )

    class TestParser(DelegatingParser):
        tool_parser_cls = Qwen3EngineToolParser

    request = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools_strict,
        tool_choice="auto",
    )
    parser = TestParser(MagicMock(), tools=sample_tools_strict)
    parser.reasoning_parser = MagicMock(adjust_request=lambda request: request)

    parser.adjust_request(request)

    assert captured == [False]


def test_xgrammar_function_parameters_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
    sample_tools_strict: list[ChatCompletionToolsParam],
):
    captured: list[list[dict]] = []

    def fake_get_xgrammar_model_structural_tag(*, tools: list[dict], **kwargs):
        captured.append(tools)
        return None

    monkeypatch.setattr(
        "vllm.tool_parsers.structural_tag_registry.get_xgrammar_model_structural_tag",
        fake_get_xgrammar_model_structural_tag,
    )

    get_model_structural_tag(
        model="llama",
        tools=sample_tools_strict,
        tool_choice="auto",
        reasoning=False,
    )

    assert (
        captured[0][0]["function"]["parameters"]
        == sample_tools_strict[0].function.parameters
    )
    assert sample_tools_strict[0].function.parameters is not None


@pytest.mark.parametrize("model", sorted(XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS))
def test_auto_tool_choice_skips_structural_tag_without_strict(
    model: str,
    sample_tools: list[ChatCompletionToolsParam],
):
    tag = get_model_structural_tag(
        model=model,
        tools=sample_tools,
        tool_choice="auto",
        reasoning=False,
    )

    assert tag is None


def test_get_function_parameters_relaxes_function_strict_false():
    function = SimpleNamespace(
        parameters={"type": "object", "properties": {}},
        strict=False,
    )

    assert _get_function_parameters(function) is True


class TestEnforceStrictToolCalling:
    """Server override behavior for structural-tag based tool calling."""

    @pytest.fixture
    def dumped_tools(self, monkeypatch: pytest.MonkeyPatch):
        """Capture the tool dicts handed to xgrammar."""
        captured: list[list[dict]] = []

        def fake_get_xgrammar_model_structural_tag(*, tools: list[dict], **kwargs):
            captured.append(tools)
            return MagicMock(spec=StructuralTag)

        monkeypatch.setattr(
            "vllm.tool_parsers.structural_tag_registry."
            "get_xgrammar_model_structural_tag",
            fake_get_xgrammar_model_structural_tag,
        )
        return captured

    @pytest.mark.parametrize(
        ("value", "request_strict", "expect_tag"),
        [
            (None, False, False),
            (None, True, True),
            ("true", False, True),
            ("false", True, False),
        ],
    )
    def test_auto_follows_explicit_server_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sample_tools: list[ChatCompletionToolsParam],
        sample_tools_strict: list[ChatCompletionToolsParam],
        value: str | None,
        request_strict: bool,
        expect_tag: bool,
    ):
        if value is None:
            monkeypatch.delenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", raising=False)
        else:
            monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", value)
        tools = sample_tools_strict if request_strict else sample_tools
        request = ChatCompletionRequest(
            messages=[], model="test-model", tools=tools, tool_choice="auto"
        )
        parser = Qwen3EngineToolParser(MagicMock(), tools=tools)

        tag = parser.get_structural_tag(request)

        assert (tag is not None) is expect_tag

    def test_force_strict_overrides_tools_without_mutating_request(
        self,
        dumped_tools: list[list[dict]],
        sample_tools: list[ChatCompletionToolsParam],
    ):
        explicitly_relaxed = ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_time",
                "strict": False,
                "parameters": {"type": "object", "properties": {}},
            },
        )
        responses_tool = FunctionTool(
            type="function",
            name="get_location",
            strict=False,
            parameters={"type": "object", "properties": {}},
        )
        tools = [*sample_tools, explicitly_relaxed, responses_tool]

        get_model_structural_tag(
            model="llama",
            tools=tools,
            tool_choice="auto",
            reasoning=False,
            force_strict=True,
        )

        assert dumped_tools[0][0]["function"]["strict"] is True
        assert dumped_tools[0][1]["function"]["strict"] is True
        assert dumped_tools[0][2]["function"]["strict"] is True
        assert sample_tools[0].function.strict is None
        assert explicitly_relaxed.function.strict is False
        assert responses_tool.strict is False

    @pytest.mark.parametrize(
        ("value", "expected_support"),
        [(None, False), ("true", False), ("false", True)],
    )
    def test_parser_routing_follows_explicit_server_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        value: str | None,
        expected_support: bool,
    ):
        if value is None:
            monkeypatch.delenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", raising=False)
        else:
            monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", value)

        class TestParser(ToolParser):
            structural_tag_model = "llama"

        assert TestParser.supports_required_and_named is expected_support

    def test_force_strict_keeps_text_response_allowed_under_auto(
        self,
        sample_tools: list[ChatCompletionToolsParam],
    ):
        auto_tag = get_model_structural_tag(
            model="hermes",
            tools=sample_tools,
            tool_choice="auto",
            reasoning=False,
            force_strict=True,
        )
        required_tag = get_model_structural_tag(
            model="hermes",
            tools=sample_tools,
            tool_choice="required",
            reasoning=False,
            force_strict=True,
        )

        assert auto_tag is not None
        assert required_tag is not None
        assert isinstance(auto_tag.format, TriggeredTagsFormat)
        assert isinstance(required_tag.format, TagsWithSeparatorFormat)

    def test_force_strict_supports_parallel_tool_calls(
        self,
        sample_tools: list[ChatCompletionToolsParam],
    ):
        tools = sample_tools + [
            ChatCompletionToolsParam(
                type="function",
                function={
                    "name": "get_time",
                    "parameters": {
                        "type": "object",
                        "properties": {"tz": {"type": "string"}},
                    },
                },
            )
        ]

        tag = get_model_structural_tag(
            model="hermes",
            tools=tools,
            tool_choice="auto",
            reasoning=False,
            force_strict=True,
        )

        assert isinstance(tag, StructuralTag)
        rendered = str(tag)
        assert "get_weather" in rendered
        assert "get_time" in rendered

    def test_force_strict_compiles_hermes_tool_call_bridge(self):
        """Hermes's captured open arguments object compiles for DeepSeek V4."""
        tool = ChatCompletionToolsParam(
            type="function",
            function={
                "name": "tool_call",
                "description": (
                    "Invoke a deferred tool by name with the given arguments. "
                    "Argument shape matches the tool's schema (see `tool_describe`). "
                    "Policy, hooks, and approvals run exactly as for any "
                    "directly-listed tool."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Exact tool name to invoke.",
                        },
                        "arguments": {
                            "type": "object",
                            "description": (
                                "Arguments for the tool, matching its schema."
                            ),
                        },
                    },
                    "required": ["name", "arguments"],
                },
            },
        )
        request = ChatCompletionRequest(
            messages=[], model="pennyroyal", tools=[tool], tool_choice="auto"
        )
        parser = DeepSeekV4ToolParser.__new__(DeepSeekV4ToolParser)

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("VLLM_ENFORCE_STRICT_TOOL_CALLING", "true")
            tag = parser.get_structural_tag(request)

        assert isinstance(tag, StructuralTag)
        assert isinstance(tag.format, TriggeredTagsFormat)
        assert Grammar.from_structural_tag(tag) is not None
        assert tool.function.strict is None
        assert tool.function.parameters is not None
        arguments = tool.function.parameters["properties"]["arguments"]
        assert arguments == {
            "type": "object",
            "description": "Arguments for the tool, matching its schema.",
        }
