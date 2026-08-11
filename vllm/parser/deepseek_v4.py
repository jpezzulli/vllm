# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 parser: ``<think>``/``</think>``
reasoning plus DSML tool calls in a single state machine.

DeepSeek V4 output format::

    <think>
    ...reasoning...
    </think>
    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="func_name">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="count" string="false">5</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
"""

from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING

import regex as re

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
from vllm.tool_parsers.utils import find_tool_properties

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

_DSML = "｜DSML｜"

DSML_THINK_START = "<think>"
DSML_THINK_END = "</think>"
DSML_TOOL_START = f"<{_DSML}tool_calls>"
DSML_TOOL_END = f"</{_DSML}tool_calls>"
DSML_INVOKE_PREFIX = f'<{_DSML}invoke name="'
DSML_INVOKE_NAME_END = '">'
DSML_INVOKE_END = f"</{_DSML}invoke>"
DSML_PARAM_CLOSE = f"</{_DSML}parameter>"
# DeepSeek V3.2-style wrapper, recognized only to reject it as foreign
DSML_FOREIGN_TOOL_START = f"<{_DSML}function_calls>"
DSML_FOREIGN_TOOL_END = f"</{_DSML}function_calls>"

_ESCAPED_DSML = re.escape(_DSML)
_PARAM_OPEN_RE = re.compile(
    rf'<{_ESCAPED_DSML}parameter\s+name="([^"]+)"\s+string="(true|false)">'
)
_MISSING = object()


def _skip_space(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _find_unquoted_close(text: str, pos: int) -> int:
    """Find a parameter close outside a JSON string."""
    in_string = False
    escaped = False
    while pos < len(text):
        char = text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            pos += 1
            continue
        if char == '"':
            in_string = True
            pos += 1
            continue
        if text.startswith(DSML_PARAM_CLOSE, pos):
            return pos
        pos += 1
    return -1


def _decode_nonstring(value: str) -> object:
    stripped = value.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return value


def _parse_dsml_sequence(
    text: str,
    pos: int,
    *,
    partial: bool,
    expect_close: bool,
) -> tuple[dict[str, object], int, bool]:
    params: dict[str, object] = {}
    while True:
        pos = _skip_space(text, pos)
        if expect_close and text.startswith(DSML_PARAM_CLOSE, pos):
            return params, pos + len(DSML_PARAM_CLOSE), True
        if pos >= len(text):
            return params, pos, not expect_close

        match = _PARAM_OPEN_RE.match(text, pos)
        if match is None:
            return params, len(text) if partial else pos, False

        name, is_string = match.group(1), match.group(2) == "true"
        value, pos, complete = _parse_dsml_value(
            text,
            match.end(),
            is_string=is_string,
            partial=partial,
        )
        if value is not _MISSING:
            params[name] = value
        if not complete:
            return params, pos, False


def _parse_dsml_value(
    text: str,
    pos: int,
    *,
    is_string: bool,
    partial: bool,
) -> tuple[object, int, bool]:
    if is_string:
        close = text.find(DSML_PARAM_CLOSE, pos)
        if close >= 0:
            return text[pos:close], close + len(DSML_PARAM_CLOSE), True
        if partial:
            return text[pos:], len(text), False
        return _MISSING, pos, False

    nested_start = _skip_space(text, pos)
    if _PARAM_OPEN_RE.match(text, nested_start) is not None:
        return _parse_dsml_sequence(
            text,
            nested_start,
            partial=partial,
            expect_close=True,
        )

    close = _find_unquoted_close(text, pos)
    if close >= 0:
        value = _decode_nonstring(text[pos:close])
        return value, close + len(DSML_PARAM_CLOSE), True
    if partial:
        try:
            return json.loads(text[pos:].strip()), len(text), False
        except (json.JSONDecodeError, ValueError):
            pass
    return _MISSING, pos, False


def _dsml_arg_converter(raw_args: str, partial: bool) -> str:
    params, _, _ = _parse_dsml_sequence(
        raw_args,
        0,
        partial=partial,
        expect_close=False,
    )
    return json.dumps(params, ensure_ascii=False)


def _unwrap_wrapper_args(
    args_json: str,
    tools: list[Tool] | None,
    func_name: str | None,
) -> str:
    if not tools or not func_name:
        return args_json
    try:
        args = json.loads(args_json)
    except (json.JSONDecodeError, ValueError):
        return args_json
    if not isinstance(args, dict):
        return args_json
    properties = find_tool_properties(tools, func_name)
    if not properties:
        return args_json
    allowed = set(properties.keys())
    for wrapper in ("arguments", "input"):
        if set(args.keys()) != {wrapper} or wrapper in allowed:
            continue
        inner = args[wrapper]
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                return args_json
        if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
            return json.dumps(inner, ensure_ascii=False)
    return args_json


@functools.cache
def deepseek_v4_config(thinking: bool = False) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="deepseek_v4",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": DSML_TOOL_START,
            "TOOL_END": DSML_TOOL_END,
            "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
            "INVOKE_NAME_END": DSML_INVOKE_NAME_END,
            "INVOKE_END": DSML_INVOKE_END,
            "PARAM_CLOSE": DSML_PARAM_CLOSE,
            "FOREIGN_START": DSML_FOREIGN_TOOL_START,
            "FOREIGN_END": DSML_FOREIGN_TOOL_END,
        },
        token_id_terminals={
            "THINK_START": DSML_THINK_START,
            "THINK_END": DSML_THINK_END,
            "TOOL_START": DSML_TOOL_START,
            "TOOL_END": DSML_TOOL_END,
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            # Absorb a bare </think> with no prior <think>
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            # Absorb a duplicate <think> while already reasoning
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # Tool call beginning while still inside <think>
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            # Orphan invoke: at long context the model may omit the
            # <｜DSML｜tool_calls> wrapper and emit the invoke directly.
            # The invoke marker has no dedicated special token, so hold
            # events and validate the parsed name before committing.
            # Only names the request declared are accepted.
            (ParserState.CONTENT, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
                validate_tool_name=True,
            ),
            # V3.2-style function_calls wrapper is foreign to V4: pass
            # it and its contents through as plain content
            (ParserState.CONTENT, "FOREIGN_START"): Transition(
                ParserState.FOREIGN_BLOCK,
                (EventType.TEXT_CHUNK,),
            ),
            (ParserState.FOREIGN_BLOCK, "FOREIGN_END"): Transition(
                ParserState.CONTENT,
                (EventType.TEXT_CHUNK,),
            ),
            # The native wrapper always wins over an unclosed foreign
            # block, so a stray foreign start cannot disable tool
            # parsing for the rest of the response.
            (ParserState.FOREIGN_BLOCK, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "INVOKE_NAME_END"): Transition(
                ParserState.TOOL_ARGS,
                (),
            ),
            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            # Parallel tool calls
            (ParserState.TOOL_BETWEEN, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        },
        content_events={
            ParserState.CONTENT: EventType.TEXT_CHUNK,
            ParserState.REASONING: EventType.REASONING_CHUNK,
            ParserState.TOOL_NAME: EventType.TOOL_NAME,
            ParserState.TOOL_ARGS: EventType.ARG_VALUE_CHUNK,
            ParserState.FOREIGN_BLOCK: EventType.TEXT_CHUNK,
        },
        arg_converter=_dsml_arg_converter,
        arg_structural_chars=frozenset(">"),
        strip_content_whitespace_with_tools=False,
        tool_args_json=False,
    )


class DeepSeekV4Parser(ParserEngine):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.pop("chat_template_kwargs", None) or {}
        thinking = bool(
            chat_kwargs.get("thinking") or chat_kwargs.get("enable_thinking")
        )
        if "thinking" not in chat_kwargs and "enable_thinking" not in chat_kwargs:
            thinking = True
        thinking = thinking and chat_kwargs.get("reasoning_effort") != "none"
        super().__init__(
            tokenizer,
            tools,
            parser_engine_config=deepseek_v4_config(thinking=thinking),
            **kwargs,
        )
        self._arg_converter = self._convert_args

    def _convert_args(self, raw_args: str, partial: bool) -> str:
        result = _dsml_arg_converter(raw_args, partial)
        if not self._tools:
            return result
        func_name = next((s.name for s in self._tool_slots if s.args == raw_args), None)
        return _unwrap_wrapper_args(result, self._tools, func_name)
