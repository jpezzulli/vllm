# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for DeepSeekV4ToolParser."""

import json
from unittest.mock import MagicMock

import pytest
from xgrammar import StructuralTag

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.deepseekv4_tool_parser import DeepSeekV4ToolParser

MOCK_TOKENIZER = MagicMock()
MOCK_TOKENIZER.get_vocab.return_value = {}

TC_START = "<｜DSML｜tool_calls>"
TC_END = "</｜DSML｜tool_calls>"
INV_START = '<｜DSML｜invoke name="'
INV_END = "</｜DSML｜invoke>"
PARAM_START = '<｜DSML｜parameter name="'
PARAM_END = "</｜DSML｜parameter>"


@pytest.fixture
def sample_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"},
                        "state": {"type": "string", "description": "The state code"},
                        "unit": {"type": "string", "enum": ["fahrenheit", "celsius"]},
                    },
                    "required": ["city", "state"],
                },
            },
        ),
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "calculate_area",
                "description": "Calculate area of a shape",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shape": {"type": "string"},
                        "dimensions": {"type": "object"},
                        "precision": {"type": "integer"},
                    },
                },
            },
        ),
    ]


def make_parser(tools=None) -> DeepSeekV4ToolParser:
    return DeepSeekV4ToolParser(MOCK_TOKENIZER, tools=tools)


def make_request(tools=None) -> MagicMock:
    req = MagicMock()
    req.tools = tools
    req.tool_choice = "auto"
    return req


def build_tool_call(func_name: str, params: dict[str, str]) -> str:
    param_strs = "".join(
        f'{PARAM_START}{k}" string="true">{v}{PARAM_END}\n' for k, v in params.items()
    )
    return f'{TC_START}\n{INV_START}{func_name}">\n{param_strs}{INV_END}\n{TC_END}'


def stream(
    parser: DeepSeekV4ToolParser,
    full_text: str,
    chunk_size: int = 7,
    request=None,
):
    deltas = []
    previous_text = ""
    request = request or make_request(parser.tools)
    for start in range(0, len(full_text), chunk_size):
        delta_text = full_text[start : start + chunk_size]
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[1],
            request=request,
        )
        previous_text = current_text
        if delta is not None:
            deltas.append(delta)
    return deltas


def reconstruct_args(deltas, tool_index: int = 0) -> str:
    fragments = []
    for delta in deltas:
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                if (
                    tool_call.index == tool_index
                    and tool_call.function
                    and tool_call.function.arguments
                ):
                    fragments.append(tool_call.function.arguments)
    return "".join(fragments)


def test_registered():
    assert ToolParserManager.get_tool_parser("deepseek_v4") is DeepSeekV4ToolParser


def test_extract_tool_calls():
    parser = make_parser()
    model_output = "Let me check. " + build_tool_call(
        "get_weather", {"location": "Beijing", "unit": "celsius"}
    )

    result = parser.extract_tool_calls(model_output, make_request())

    assert result.tools_called
    assert result.content == "Let me check. "
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {
        "location": "Beijing",
        "unit": "celsius",
    }


def test_function_calls_block_is_not_accepted():
    parser = make_parser()
    model_output = build_tool_call("search", {"query": "vllm"}).replace(
        "tool_calls", "function_calls"
    )

    result = parser.extract_tool_calls(model_output, make_request())

    assert not result.tools_called
    assert result.content == model_output


def test_streaming_extracts_complete_invokes():
    parser = make_parser()
    full_text = build_tool_call("search", {"query": "deepseek v4"})

    deltas = stream(parser, full_text, chunk_size=5)

    names = [
        tool_call.function.name
        for delta in deltas
        if delta.tool_calls
        for tool_call in delta.tool_calls
        if tool_call.function.name
    ]
    assert names == ["search"]
    assert json.loads(reconstruct_args(deltas)) == {"query": "deepseek v4"}


@pytest.mark.parametrize("chunk_size", [1, 4, 17])
def test_streaming_emits_incremental_argument_chunks(chunk_size: int):
    tool = ChatCompletionToolsParam(
        function=FunctionDefinition(
            name="plan_trip",
            parameters={
                "type": "object",
                "properties": {
                    "days": {"type": "integer"},
                    "flexible": {"type": "boolean"},
                    "cities": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
            },
        ),
    )
    parser = make_parser(tools=[tool])
    full_text = (
        f"{TC_START}\n"
        f'{INV_START}plan_trip">\n'
        f'{PARAM_START}days" string="false">3{PARAM_END}\n'
        f'{PARAM_START}flexible" string="false">false{PARAM_END}\n'
        f'{PARAM_START}cities" string="false">'
        f'["Beijing","Shanghai","Tokyo","New York"]{PARAM_END}\n'
        f'{PARAM_START}notes" string="true">靠窗座位{PARAM_END}\n'
        f"{INV_END}\n"
        f"{TC_END}"
    )

    deltas = stream(parser, full_text, chunk_size=chunk_size)
    arg_chunks = [
        tool_call.function.arguments
        for delta in deltas
        for tool_call in delta.tool_calls or []
        if tool_call.function and tool_call.function.arguments is not None
    ]

    assert len([chunk for chunk in arg_chunks if chunk]) > 2
    assert json.loads("".join(arg_chunks)) == {
        "days": 3,
        "flexible": False,
        "cities": ["Beijing", "Shanghai", "Tokyo", "New York"],
        "notes": "靠窗座位",
    }
    content = "".join(delta.content or "" for delta in deltas)
    assert "DSML" not in content
    assert not parser._in_tool_calls
    assert parser._active_tool_index is None
    assert parser._buffer == ""


def _with_strict(
    tools: list[ChatCompletionToolsParam],
) -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            type=t.type,
            function=FunctionDefinition(
                name=t.function.name,
                description=t.function.description,
                parameters=t.function.parameters,
                strict=True,
            ),
        )
        for t in tools
    ]


def test_get_vllm_registry_structural_tag_returns_structural_tag(
    sample_tools: list[ChatCompletionToolsParam],
) -> None:
    parser = make_parser()
    strict_tools = _with_strict(sample_tools)
    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=strict_tools,
        tool_choice="auto",
    )
    tag = parser.get_structural_tag(req)
    assert isinstance(tag, StructuralTag)

    req = ChatCompletionRequest(
        messages=[],
        model="m",
        tools=sample_tools,
        tool_choice="required",
    )
    tag = parser.get_structural_tag(req)
    assert isinstance(tag, StructuralTag)

    if sample_tools:
        tool = sample_tools[0]
        req = ChatCompletionRequest(
            messages=[],
            model="m",
            tools=sample_tools,
        )
        req.tool_choice = ChatCompletionNamedToolChoiceParam(
            function=ChatCompletionNamedFunction(name=tool.function.name)
        )
        tag = parser.get_structural_tag(req)
        assert isinstance(tag, StructuralTag)


def test_extract_tool_calls_arguments_wrapper():
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {}

    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        },
    )

    parser = DeepSeekV4ToolParser(mock_tokenizer, tools=[tool])
    request = MagicMock()
    request.tools = [tool]

    model_output = (
        f"{TC_START}"
        f'{INV_START}get_weather">'
        f'{PARAM_START}arguments" string="false">{{"location":"Beijing"}}{PARAM_END}'
        f"{INV_END}"
        f"{TC_END}"
    )

    result = parser.extract_tool_calls(model_output, request)
    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {"location": "Beijing"}


def test_string_attribute_and_guarded_wrapper_semantics():
    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "record_values",
            "parameters": {
                "type": "object",
                "properties": {
                    "literal": {"type": "integer"},
                    "coerced": {"type": "integer"},
                    "arguments": {"type": "object"},
                },
            },
        },
    )
    parser = make_parser([tool])
    output = (
        f'{TC_START}{INV_START}record_values">'
        f'{PARAM_START}literal" string="true">007{PARAM_END}'
        f'{PARAM_START}coerced" string="false">7{PARAM_END}'
        f'{PARAM_START}arguments" string="false">'
        '{"nested":{"enabled":true}}'
        f"{PARAM_END}"
        f"{INV_END}{TC_END}"
    )

    result = parser.extract_tool_calls(output, make_request([tool]))

    assert json.loads(result.tool_calls[0].function.arguments) == {
        "literal": "007",
        "coerced": 7,
        "arguments": {"nested": {"enabled": True}},
    }


def test_guarded_artificial_wrapper_rejects_unknown_inner_keys():
    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        },
    )
    parser = make_parser([tool])
    output = (
        f'{TC_START}{INV_START}get_weather">'
        f'{PARAM_START}arguments" string="false">'
        '{"unknown":"Beijing"}'
        f"{PARAM_END}{INV_END}{TC_END}"
    )

    result = parser.extract_tool_calls(output, make_request([tool]))

    assert json.loads(result.tool_calls[0].function.arguments) == {
        "arguments": '{"unknown":"Beijing"}'
    }


def _image_tool() -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": "image_generate",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "options": {
                        "type": "object",
                        "properties": {
                            "size": {
                                "type": "object",
                                "properties": {
                                    "width": {"type": "integer"},
                                    "height": {"type": "integer"},
                                },
                            },
                            "seed": {"type": "integer"},
                        },
                    },
                },
            },
        },
    )


def _terminal_tool() -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": "terminal_exec",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "environment": {"type": "object"},
                },
            },
        },
    )


def _image_invoke() -> str:
    return (
        f'{INV_START}image_generate">'
        f'{PARAM_START}prompt" string="true">draw a copper owl{PARAM_END}'
        f'{PARAM_START}options" string="false">'
        '{"size":{"width":1024,"height":768},"seed":731}'
        f"{PARAM_END}{INV_END}"
    )


def test_orphan_invoke_recovers_declared_nested_image_tool_non_streaming():
    tool = _image_tool()
    output = _image_invoke() + TC_END + "\nImage queued."
    parser = make_parser([tool])

    result = parser.extract_tool_calls(output, make_request([tool]))

    assert result.tools_called
    assert result.content == "\nImage queued."
    assert result.tool_calls[0].function.name == "image_generate"
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "prompt": "draw a copper owl",
        "options": {"size": {"width": 1024, "height": 768}, "seed": 731},
    }


def test_orphan_invoke_streams_without_control_text_leakage():
    tool = _image_tool()
    parser = make_parser([tool])
    output = _image_invoke() + TC_END

    deltas = stream(parser, output, chunk_size=1, request=make_request([tool]))

    names = [
        call.function.name
        for delta in deltas
        for call in delta.tool_calls or []
        if call.function and call.function.name
    ]
    arguments = reconstruct_args(deltas)
    content = "".join(delta.content or "" for delta in deltas)
    assert names == ["image_generate"]
    assert json.loads(arguments)["options"]["size"] == {
        "width": 1024,
        "height": 768,
    }
    assert "DSML" not in content
    assert "R0TURN" not in content
    assert not parser._in_tool_calls
    assert parser._active_tool_index is None
    assert parser._buffer == ""


@pytest.mark.parametrize("mode", ["unknown", "no-tools", "tool-choice-none"])
def test_false_orphan_invoke_stays_content(mode: str):
    tool = _image_tool()
    request = make_request([tool])
    output = _image_invoke()
    if mode == "unknown":
        output = output.replace("image_generate", "unknown_image_tool", 1)
    elif mode == "no-tools":
        request.tools = []
    else:
        request.tool_choice = "none"
    parser = make_parser([tool])

    result = parser.extract_tool_calls(output, request)

    assert not result.tools_called
    assert result.tool_calls == []
    assert result.content == output


@pytest.mark.parametrize("mode", ["unknown", "no-tools", "tool-choice-none"])
def test_false_orphan_invoke_streaming_stays_content(mode: str):
    tool = _image_tool()
    request = make_request([tool])
    output = _image_invoke()
    if mode == "unknown":
        output = output.replace("image_generate", "unknown_image_tool", 1)
    elif mode == "no-tools":
        request.tools = []
    else:
        request.tool_choice = "none"
    parser = make_parser([tool])

    deltas = stream(parser, output, chunk_size=1, request=request)

    assert reconstruct_args(deltas) == ""
    assert "".join(delta.content or "" for delta in deltas) == output


def test_quoted_marker_then_real_wrapped_terminal_call():
    tool = _terminal_tool()
    request = make_request([tool])
    quoted = f"Documentation quotes {INV_START} literally. "
    invoke = (
        f'{INV_START}terminal_exec">'
        f'{PARAM_START}command" string="true">printf ready{PARAM_END}'
        f'{PARAM_START}environment" string="false">'
        '{"MODE":"safe","FLAGS":{"trace":false}}'
        f"{PARAM_END}"
        f"{INV_END}"
    )
    output = quoted + TC_START + invoke + TC_END

    result = make_parser([tool]).extract_tool_calls(output, request)
    parser = make_parser([tool])
    deltas = stream(parser, output, chunk_size=1, request=request)

    assert result.tools_called
    assert result.content == quoted
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "command": "printf ready",
        "environment": {"MODE": "safe", "FLAGS": {"trace": False}},
    }
    assert json.loads(reconstruct_args(deltas)) == json.loads(
        result.tool_calls[0].function.arguments
    )
    streamed_content = "".join(delta.content or "" for delta in deltas)
    assert streamed_content == quoted


def test_foreign_wrapper_is_preserved_as_content():
    tool = _image_tool()
    foreign = (
        "<｜DSML｜function_calls>"
        + _image_invoke()
        + "</｜DSML｜function_calls>"
    )

    result = make_parser([tool]).extract_tool_calls(foreign, make_request([tool]))
    parser = make_parser([tool])
    deltas = stream(parser, foreign, chunk_size=1, request=make_request([tool]))

    assert not result.tools_called
    assert result.content == foreign
    assert reconstruct_args(deltas) == ""
    assert "".join(delta.content or "" for delta in deltas) == foreign


def test_declared_names_do_not_leak_between_streams():
    tool = _image_tool()
    parser = make_parser([tool])
    first = stream(parser, _image_invoke(), 3, make_request([tool]))
    assert json.loads(reconstruct_args(first))["prompt"] == "draw a copper owl"

    second = stream(parser, _image_invoke(), 3, make_request([]))

    assert reconstruct_args(second) == ""
    assert "".join(delta.content or "" for delta in second) == _image_invoke()

    non_streaming = make_parser([tool])
    first_result = non_streaming.extract_tool_calls(
        _image_invoke(), make_request([tool])
    )
    second_result = non_streaming.extract_tool_calls(_image_invoke(), make_request([]))
    assert first_result.tools_called
    assert not second_result.tools_called
    assert second_result.content == _image_invoke()


@pytest.mark.skip_global_cleanup
def test_composed_schema_converts_object_and_array_params():
    tool = ChatCompletionToolsParam(
        type="function",
        function={
            "name": "set_timer",
            "parameters": {
                "type": "object",
                "properties": {
                    "wait": {
                        "anyOf": [
                            {"type": "object"},
                            {"type": "null"},
                        ],
                    },
                    "patches": {
                        "allOf": [
                            {"type": "array", "items": {"type": "object"}},
                        ],
                    },
                },
            },
        },
    )
    parser = make_parser(tools=[tool])
    request = make_request(tools=[tool])
    model_output = (
        f"{TC_START}\n"
        f'{INV_START}set_timer">\n'
        f'{PARAM_START}wait" string="false">'
        f'{{"type":"for","minutes":2880}}'
        f"{PARAM_END}\n"
        f'{PARAM_START}patches" string="false">'
        f'[{{"op":"replace","path":"/schedule","value":"quiet"}}]'
        f"{PARAM_END}\n"
        f"{INV_END}\n"
        f"{TC_END}"
    )

    result = parser.extract_tool_calls(model_output, request)

    assert result.tools_called
    args = json.loads(result.tool_calls[0].function.arguments)
    assert args == {
        "wait": {"type": "for", "minutes": 2880},
        "patches": [{"op": "replace", "path": "/schedule", "value": "quiet"}],
    }
