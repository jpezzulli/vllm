# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import uuid
from collections.abc import Sequence
from typing import Any, Literal

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.utils import (
    coerce_to_schema_type,
    collect_tool_names,
    extract_types_from_schema,
    find_tool_properties,
    partial_tag_overlap,
)

logger = init_logger(__name__)


class DeepSeekV32ToolParser(ToolParser):
    """
    example tool call content:
    <｜DSML｜function_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
    </｜DSML｜invoke>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">北京</｜DSML｜parameter>
    <｜DSML｜parameter name="date" string="true">2024-01-16</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    """

    tool_call_start_token: str = "<｜DSML｜function_calls>"
    tool_call_end_token: str = "</｜DSML｜function_calls>"
    foreign_tool_call_start_token: str = "<｜DSML｜tool_calls>"
    foreign_tool_call_end_token: str = "</｜DSML｜tool_calls>"
    invoke_prefix_token: str = '<｜DSML｜invoke name="'
    invoke_name_end_token: str = '">'
    structural_tag_model = "deepseek_v3_2"

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.prev_tool_call_arr: list[dict] = []

        # Streaming state
        self.current_tool_index: int = 0
        self._sent_content_idx: int = 0
        self._buffer: str = ""
        self._in_tool_calls: bool = False
        self._active_tool_index: int | None = None
        self._active_tool_name: str | None = None
        self._active_param_name: str | None = None
        self._active_param_string_attr: str | None = None
        self._active_param_mode: str | None = None
        self._active_param_parts: list[str] = []
        self._args_started: list[bool] = []
        self._allowed_tool_names: frozenset[str] = frozenset()
        self._orphan_tool_calls: bool = False
        self._in_foreign_tool_calls: bool = False
        self._after_orphan: bool = False

        # Regex patterns for complete parsing
        self.tool_call_complete_regex = re.compile(
            re.escape(self.tool_call_start_token)
            + r"(.*?)"
            + re.escape(self.tool_call_end_token),
            re.DOTALL,
        )
        self.invoke_complete_regex = re.compile(
            r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>', re.DOTALL
        )
        self.parameter_complete_regex = re.compile(
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>(.*?)</｜DSML｜parameter>',
            re.DOTALL,
        )
        self.invoke_start_regex = re.compile(r'<｜DSML｜invoke\s+name="([^"]+)"\s*>')
        self.parameter_start_regex = re.compile(
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="(true|false)"\s*>'
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        logger.debug(
            "vLLM Successfully import tool parser %s !", self.__class__.__name__
        )

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Ensure tool call tokens
            # (e.g. <｜DSML｜function_calls>, </｜DSML｜function_calls>)
            # are not skippedduring decoding.
            # Even though they are not marked as special tokens,
            # setting skip_special_tokens=False ensures proper handling in
            # transformers 5.x where decoding behavior may have changed.
            request.skip_special_tokens = False
        return request

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _prepare_request(self, request: ChatCompletionRequest) -> None:
        tools = getattr(request, "tools", None)
        if tools is not None:
            self.tools = tools
        if tools and getattr(request, "tool_choice", None) != "none":
            self._allowed_tool_names = collect_tool_names(tools)
        else:
            self._allowed_tool_names = frozenset()

    def _parse_invoke_params(self, invoke_str: str) -> dict[str, tuple[str, str]]:
        param_dict: dict[str, tuple[str, str]] = {}
        for param_name, string_attr, param_val in self.parameter_complete_regex.findall(
            invoke_str
        ):
            param_dict[param_name] = (param_val, string_attr)
        return param_dict

    @staticmethod
    def _repair_param_dict(
        param_dict: dict[str, Any],
        param_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Unwrap single 'arguments' / 'input' wrappers when the wrapper
        is not part of the requested tool schema and the wrapped object
        matches the schema fields."""
        allowed = set(param_config.keys())
        for wrapper in ("arguments", "input"):
            if set(param_dict.keys()) != {wrapper} or wrapper in allowed:
                continue
            inner = param_dict[wrapper]
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except json.JSONDecodeError:
                    return param_dict
            if isinstance(inner, dict) and set(inner.keys()).issubset(allowed):
                return inner
        return param_dict

    def _convert_params_with_schema(
        self,
        function_name: str,
        param_dict: dict[str, tuple[str, str]],
    ) -> dict[str, Any]:
        """Convert raw string param values using the tool schema types."""
        param_config = find_tool_properties(self.tools, function_name)

        converted: dict[str, Any] = {}
        for name, (value, string_attr) in param_dict.items():
            if string_attr == "true":
                converted[name] = value
                continue

            param_types = extract_types_from_schema(param_config.get(name, {}))
            converted[name] = coerce_to_schema_type(value, param_types)
        return self._repair_param_dict(converted, param_config)

    def _get_param_config(self, function_name: str | None) -> dict[str, Any]:
        if not function_name or not self.tools:
            return {}
        return find_tool_properties(self.tools, function_name)

    @staticmethod
    def _json_escape_string_content(text: str) -> str:
        return json.dumps(text, ensure_ascii=False)[1:-1]

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete model output (non-streaming)."""
        self._prepare_request(request)

        if self.invoke_prefix_token in model_output:
            return self._extract_tool_calls_with_orphan_recovery(model_output)

        # Quick check
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            tool_calls = []

            # Find all complete tool_call blocks
            for tool_call_match in self.tool_call_complete_regex.findall(model_output):
                # Find all invokes within this tool_call
                for invoke_name, invoke_content in self.invoke_complete_regex.findall(
                    tool_call_match
                ):
                    tool_calls.append(
                        self._make_tool_call(invoke_name, invoke_content)
                    )

            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            # Extract content before first tool call
            first_tool_idx = model_output.find(self.tool_call_start_token)
            content = model_output[:first_tool_idx] if first_tool_idx > 0 else None

            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=content
            )

        except Exception:
            logger.exception("Error extracting tool calls")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _make_tool_call(self, invoke_name: str, invoke_content: str) -> ToolCall:
        param_dict = self._parse_invoke_params(invoke_content)
        params = self._convert_params_with_schema(invoke_name, param_dict)
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=invoke_name,
                arguments=json.dumps(params, ensure_ascii=False),
            ),
        )

    def _extract_tool_calls_with_orphan_recovery(
        self, model_output: str
    ) -> ExtractedToolCallInformation:
        """Adapt the orphan-invoke recovery from vLLM PR #49117."""
        tool_calls: list[ToolCall] = []
        content_parts: list[str] = []
        cursor = 0

        while cursor < len(model_output):
            candidates = [
                (model_output.find(self.tool_call_start_token, cursor), "native"),
                (
                    model_output.find(self.foreign_tool_call_start_token, cursor),
                    "foreign",
                ),
            ]
            if self._allowed_tool_names:
                candidates.append(
                    (model_output.find(self.invoke_prefix_token, cursor), "orphan")
                )
            candidates = [(idx, kind) for idx, kind in candidates if idx >= 0]
            if not candidates:
                content_parts.append(model_output[cursor:])
                break

            marker_idx, marker_kind = min(candidates, key=lambda item: item[0])
            content_parts.append(model_output[cursor:marker_idx])

            if marker_kind == "foreign":
                foreign_end = model_output.find(
                    self.foreign_tool_call_end_token,
                    marker_idx + len(self.foreign_tool_call_start_token),
                )
                native_inside = model_output.find(
                    self.tool_call_start_token,
                    marker_idx + len(self.foreign_tool_call_start_token),
                )
                if native_inside >= 0 and (
                    foreign_end < 0 or native_inside < foreign_end
                ):
                    content_parts.append(model_output[marker_idx:native_inside])
                    cursor = native_inside
                    continue
                if foreign_end < 0:
                    content_parts.append(model_output[marker_idx:])
                    break
                foreign_after = foreign_end + len(self.foreign_tool_call_end_token)
                content_parts.append(model_output[marker_idx:foreign_after])
                cursor = foreign_after
                continue

            if marker_kind == "native":
                block_end = model_output.find(
                    self.tool_call_end_token,
                    marker_idx + len(self.tool_call_start_token),
                )
                if block_end < 0:
                    content_parts.append(model_output[marker_idx:])
                    break
                block_after = block_end + len(self.tool_call_end_token)
                block = model_output[
                    marker_idx + len(self.tool_call_start_token) : block_end
                ]
                parsed = [
                    self._make_tool_call(name, invoke_content)
                    for name, invoke_content in (
                        self.invoke_complete_regex.findall(block)
                    )
                ]
                if parsed:
                    tool_calls.extend(parsed)
                else:
                    content_parts.append(model_output[marker_idx:block_after])
                cursor = block_after
                continue

            match = self.invoke_complete_regex.match(model_output, marker_idx)
            if match is None or match.group(1) not in self._allowed_tool_names:
                content_parts.append(model_output[marker_idx])
                cursor = marker_idx + 1
                continue
            tool_calls.append(self._make_tool_call(match.group(1), match.group(2)))
            cursor = match.end()
            closing = re.match(
                r"\s*" + re.escape(self.tool_call_end_token), model_output[cursor:]
            )
            if closing is not None:
                cursor += closing.end()

        if not tool_calls:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        content = "".join(content_parts)
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content if content.strip() else None,
        )

    def _reset_streaming_state(self):
        """Reset all streaming state."""
        self.current_tool_index = 0
        self._sent_content_idx = 0
        self._buffer = ""
        self._in_tool_calls = False
        self._active_tool_index = None
        self._active_tool_name = None
        self._active_param_name = None
        self._active_param_string_attr = None
        self._active_param_mode = None
        self._active_param_parts.clear()
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()
        self._args_started.clear()
        self._allowed_tool_names = frozenset()
        self._orphan_tool_calls = False
        self._in_foreign_tool_calls = False
        self._after_orphan = False

    @staticmethod
    def _first_marker(text: str, markers: list[str]) -> tuple[int, str] | None:
        found = [(text.find(marker), marker) for marker in markers]
        found = [(idx, marker) for idx, marker in found if idx >= 0]
        return min(found, default=None, key=lambda item: item[0])

    def _can_grow_into_allowed_name(self, name_text: str) -> bool:
        candidates = [name_text]
        for size in range(1, len(self.invoke_name_end_token)):
            if name_text.endswith(self.invoke_name_end_token[:size]):
                candidates.append(name_text[:-size])
        return any(
            allowed.startswith(candidate)
            for allowed in self._allowed_tool_names
            for candidate in candidates
        )

    def _process_content_buffer(
        self,
        content_parts: list[str],
        tool_call_deltas: dict[int, DeltaToolCall],
    ) -> bool:
        markers = [self.tool_call_start_token, self.foreign_tool_call_start_token]
        if self._allowed_tool_names:
            markers.append(self.invoke_prefix_token)

        if self._after_orphan:
            stripped = self._buffer.lstrip()
            if not stripped:
                return False
            if stripped.startswith(self.tool_call_end_token):
                self._buffer = stripped[len(self.tool_call_end_token) :]
                self._after_orphan = False
                return True
            if self.tool_call_end_token.startswith(stripped):
                return False
            recoverable = [self.tool_call_start_token, self.invoke_prefix_token]
            if any(
                marker.startswith(stripped) or stripped.startswith(marker)
                for marker in recoverable
            ):
                self._buffer = stripped
                self._after_orphan = False
                return True
            content_parts.append(self._buffer)
            self._buffer = ""
            self._after_orphan = False
            return False

        if self._in_foreign_tool_calls:
            foreign_markers = [
                self.foreign_tool_call_end_token,
                self.tool_call_start_token,
            ]
            found = self._first_marker(self._buffer, foreign_markers)
            if found is None:
                overlap = max(
                    partial_tag_overlap(self._buffer, marker)
                    for marker in foreign_markers
                )
                safe_len = len(self._buffer) - overlap
                if safe_len > 0:
                    content_parts.append(self._buffer[:safe_len])
                    self._buffer = self._buffer[safe_len:]
                return False
            marker_idx, marker = found
            if marker_idx > 0:
                content_parts.append(self._buffer[:marker_idx])
                self._buffer = self._buffer[marker_idx:]
            if marker == self.tool_call_start_token:
                self._buffer = self._buffer[len(marker) :]
                self._in_foreign_tool_calls = False
                self._in_tool_calls = True
                return True
            content_parts.append(marker)
            self._buffer = self._buffer[len(marker) :]
            self._in_foreign_tool_calls = False
            return True

        found = self._first_marker(self._buffer, markers)
        if found is None:
            overlap = max(
                partial_tag_overlap(self._buffer, marker) for marker in markers
            )
            safe_len = len(self._buffer) - overlap
            if safe_len > 0:
                content_parts.append(self._buffer[:safe_len])
                self._buffer = self._buffer[safe_len:]
            return False

        marker_idx, marker = found
        if marker_idx > 0:
            content_parts.append(self._buffer[:marker_idx])
            self._buffer = self._buffer[marker_idx:]
            return True

        if marker == self.tool_call_start_token:
            self._buffer = self._buffer[len(marker) :]
            self._in_tool_calls = True
            self._orphan_tool_calls = False
            return True
        if marker == self.foreign_tool_call_start_token:
            content_parts.append(marker)
            self._buffer = self._buffer[len(marker) :]
            self._in_foreign_tool_calls = True
            return True

        name_text = self._buffer[len(self.invoke_prefix_token) :]
        native_idx = name_text.find(self.tool_call_start_token)
        name_end = name_text.find(self.invoke_name_end_token)
        if native_idx >= 0 and (name_end < 0 or native_idx < name_end):
            content_parts.append(
                self._buffer[: len(self.invoke_prefix_token) + native_idx]
            )
            self._buffer = name_text[native_idx:]
            return True
        if name_end < 0:
            if self._can_grow_into_allowed_name(name_text):
                return False
            content_parts.append(self._buffer[0])
            self._buffer = self._buffer[1:]
            return True

        name = name_text[:name_end]
        if name not in self._allowed_tool_names:
            content_parts.append(self._buffer[0])
            self._buffer = self._buffer[1:]
            return True

        self._buffer = name_text[name_end + len(self.invoke_name_end_token) :]
        self._in_tool_calls = True
        self._orphan_tool_calls = True
        self._begin_streaming_tool_call(name, tool_call_deltas)
        return True

    def _add_tool_call_delta(
        self,
        tool_call_deltas: dict[int, DeltaToolCall],
        index: int,
        *,
        call_id: str | None = None,
        call_type: Literal["function"] | None = None,
        name: str | None = None,
        arguments: str | None = None,
    ) -> None:
        if arguments:
            self.streamed_args_for_tool[index] += arguments

        if index not in tool_call_deltas:
            tool_call_deltas[index] = DeltaToolCall(
                index=index,
                id=call_id,
                type=call_type,
                function=DeltaFunctionCall(name=name, arguments=arguments),
            )
            return

        delta = tool_call_deltas[index]
        if call_id is not None:
            delta.id = call_id
        if call_type is not None:
            delta.type = call_type
        if delta.function is None:
            delta.function = DeltaFunctionCall()
        if name is not None:
            delta.function.name = name
        if arguments is not None:
            delta.function.arguments = (delta.function.arguments or "") + arguments

    def _begin_streaming_tool_call(
        self,
        name: str,
        tool_call_deltas: dict[int, DeltaToolCall],
    ) -> None:
        index = self.current_tool_index
        self.current_tool_index += 1
        self._active_tool_index = index
        self._active_tool_name = name
        self.prev_tool_call_arr.append({"name": name, "arguments": {}})
        self.streamed_args_for_tool.append("")
        self._args_started.append(False)
        self._add_tool_call_delta(
            tool_call_deltas,
            index,
            call_id=self._generate_tool_call_id(),
            call_type="function",
            name=name,
            arguments="",
        )

    def _append_param_prefix(
        self,
        tool_call_deltas: dict[int, DeltaToolCall],
        index: int,
        key: str,
        *,
        as_string: bool,
    ) -> None:
        prefix = "{" if not self._args_started[index] else ","
        self._args_started[index] = True
        arguments = prefix + json.dumps(key, ensure_ascii=False) + ":"
        if as_string:
            arguments += '"'
        self._add_tool_call_delta(tool_call_deltas, index, arguments=arguments)

    def _append_json_param_value(
        self,
        tool_call_deltas: dict[int, DeltaToolCall],
        index: int,
        key: str,
        value: Any,
    ) -> None:
        self._append_param_prefix(tool_call_deltas, index, key, as_string=False)
        self._add_tool_call_delta(
            tool_call_deltas,
            index,
            arguments=json.dumps(value, ensure_ascii=False),
        )

    def _param_types_for_name(self, name: str) -> list[str]:
        param_config = self._get_param_config(self._active_tool_name)
        if name in param_config and isinstance(param_config[name], dict):
            return extract_types_from_schema(param_config[name])
        return ["string"]

    @staticmethod
    def _can_stream_raw_param(param_types: list[str]) -> bool:
        # Scalars and unions need the complete value so streaming and
        # non-streaming share the same coercion fallback behavior.
        return set(param_types).issubset({"object", "array"})

    def _should_buffer_wrapper_param(self, name: str) -> bool:
        if (
            self._active_tool_index is None
            or self._args_started[self._active_tool_index]
        ):
            return False
        param_config = self._get_param_config(self._active_tool_name)
        return bool(
            param_config and name in ("arguments", "input") and name not in param_config
        )

    def _finish_buffered_param(
        self,
        tool_call_deltas: dict[int, DeltaToolCall],
        index: int,
    ) -> None:
        assert self._active_param_name is not None
        assert self._active_param_string_attr is not None
        raw_value = "".join(self._active_param_parts)
        converted = self._convert_params_with_schema(
            self._active_tool_name or "",
            {self._active_param_name: (raw_value, self._active_param_string_attr)},
        )
        for key, value in converted.items():
            self._append_json_param_value(tool_call_deltas, index, key, value)

    def _close_streaming_tool_call(
        self,
        tool_call_deltas: dict[int, DeltaToolCall],
    ) -> None:
        index = self._active_tool_index
        if index is None:
            return

        suffix = "}" if self._args_started[index] else "{}"
        self._add_tool_call_delta(tool_call_deltas, index, arguments=suffix)
        try:
            self.prev_tool_call_arr[index] = {
                "name": self._active_tool_name,
                "arguments": json.loads(self.streamed_args_for_tool[index]),
            }
        except (json.JSONDecodeError, IndexError):
            logger.exception("Failed to finalize DeepSeek DSML streaming tool call")

        self._active_tool_index = None
        self._active_tool_name = None
        self._active_param_name = None
        self._active_param_string_attr = None
        self._active_param_mode = None
        self._active_param_parts.clear()

    def _process_streaming_buffer(
        self,
        content_parts: list[str],
        tool_call_deltas: dict[int, DeltaToolCall],
    ) -> None:
        parameter_end_token = "</｜DSML｜parameter>"
        invoke_end_token = "</｜DSML｜invoke>"

        while True:
            if not self._in_tool_calls:
                if self._process_content_buffer(content_parts, tool_call_deltas):
                    continue
                return

            if self._active_tool_index is None:
                stripped_len = len(self._buffer) - len(self._buffer.lstrip())
                if stripped_len:
                    self._buffer = self._buffer[stripped_len:]
                    continue

                if self._buffer.startswith(self.tool_call_end_token):
                    self._buffer = self._buffer[len(self.tool_call_end_token) :]
                    self._in_tool_calls = False
                    continue

                match = self.invoke_start_regex.match(self._buffer)
                if match is None:
                    return

                self._buffer = self._buffer[match.end() :]
                self._begin_streaming_tool_call(match.group(1), tool_call_deltas)
                continue

            index = self._active_tool_index

            if self._active_param_mode is not None:
                end_pos = self._buffer.find(parameter_end_token)
                if end_pos != -1:
                    raw_content = self._buffer[:end_pos]
                    self._buffer = self._buffer[end_pos + len(parameter_end_token) :]
                    if self._active_param_mode in ("wrapper", "buffered"):
                        self._active_param_parts.append(raw_content)
                        self._finish_buffered_param(tool_call_deltas, index)
                    elif self._active_param_mode == "string":
                        arguments = self._json_escape_string_content(raw_content) + '"'
                        self._add_tool_call_delta(
                            tool_call_deltas, index, arguments=arguments
                        )
                    else:
                        self._add_tool_call_delta(
                            tool_call_deltas, index, arguments=raw_content
                        )

                    self._active_param_name = None
                    self._active_param_string_attr = None
                    self._active_param_mode = None
                    self._active_param_parts.clear()
                    continue

                overlap = partial_tag_overlap(self._buffer, parameter_end_token)
                safe_len = len(self._buffer) - overlap
                if safe_len > 0:
                    raw_content = self._buffer[:safe_len]
                    self._buffer = self._buffer[safe_len:]
                    if self._active_param_mode in ("wrapper", "buffered"):
                        self._active_param_parts.append(raw_content)
                    elif self._active_param_mode == "string":
                        self._add_tool_call_delta(
                            tool_call_deltas,
                            index,
                            arguments=self._json_escape_string_content(raw_content),
                        )
                    else:
                        self._add_tool_call_delta(
                            tool_call_deltas, index, arguments=raw_content
                        )
                return

            stripped_len = len(self._buffer) - len(self._buffer.lstrip())
            if stripped_len:
                self._buffer = self._buffer[stripped_len:]
                continue

            if self._buffer.startswith(invoke_end_token):
                self._buffer = self._buffer[len(invoke_end_token) :]
                was_orphan = self._orphan_tool_calls
                self._close_streaming_tool_call(tool_call_deltas)
                if was_orphan:
                    self._in_tool_calls = False
                    self._orphan_tool_calls = False
                    self._after_orphan = True
                continue

            match = self.parameter_start_regex.match(self._buffer)
            if match is None:
                return

            self._buffer = self._buffer[match.end() :]
            name = match.group(1)
            string_attr = match.group(2)
            self._active_param_name = name
            self._active_param_string_attr = string_attr

            if self._should_buffer_wrapper_param(name):
                self._active_param_mode = "wrapper"
                continue

            if string_attr == "true":
                self._append_param_prefix(tool_call_deltas, index, name, as_string=True)
                self._active_param_mode = "string"
                continue

            param_types = self._param_types_for_name(name)
            if not self._can_stream_raw_param(param_types):
                self._active_param_mode = "buffered"
                continue

            self._append_param_prefix(tool_call_deltas, index, name, as_string=False)
            self._active_param_mode = "raw"

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],  # pylint: disable=unused-argument
        current_token_ids: Sequence[int],  # pylint: disable=unused-argument
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """Extract tool calls from streaming model output.

        Buffers DSML markup while streaming tool-call metadata and argument
        JSON fragments as soon as they are complete enough to be valid deltas.
        """

        # First chunk of a new stream — reset state from prior request.
        if not previous_text:
            self._reset_streaming_state()
            self._prepare_request(request)

        self._buffer += delta_text
        content_parts: list[str] = []
        tool_call_deltas: dict[int, DeltaToolCall] = {}
        self._process_streaming_buffer(content_parts, tool_call_deltas)

        if not delta_text and self._buffer and not self._in_tool_calls:
            if not self._after_orphan:
                content_parts.append(self._buffer)
            self._buffer = ""

        if content_parts or tool_call_deltas:
            content = "".join(content_parts) or None
            return DeltaMessage(
                content=content, tool_calls=list(tool_call_deltas.values())
            )

        # Empty delta with token ids means EOS or closing tag; return
        # non-None so the serving framework can finalize finish_reason.
        if not delta_text and delta_token_ids and self.prev_tool_call_arr:
            return DeltaMessage(content="")

        return None
