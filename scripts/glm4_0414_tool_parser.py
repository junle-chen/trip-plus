"""vLLM tool parser for zai-org/GLM-4-32B-0414.

GLM-4-32B-0414's official chat template emits native tool calls as:

    <|assistant|>tool_name
    {"arg": "value"}

This parser converts that native format into OpenAI-compatible tool_calls.
"""

import ast
import json
import re
from collections.abc import Sequence
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParser


@ToolParserManager.register_module("glm4_0414")
class Glm40414ToolParser(ToolParser):
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        tool_names = self._tool_names(getattr(request, "tools", None))
        if not model_output or not tool_names:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        calls = []
        for chunk in self._candidate_chunks(model_output):
            parsed = self._parse_chunk(chunk, tool_names)
            if parsed is None:
                continue
            name, arguments = parsed
            calls.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(name=name, arguments=arguments),
                )
            )

        return ExtractedToolCallInformation(
            tools_called=bool(calls),
            tool_calls=calls,
            content=None if calls else model_output,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        return None

    @staticmethod
    def _tool_names(tools: Any) -> set[str]:
        names: set[str] = set()
        for tool in tools or []:
            function = getattr(tool, "function", None)
            if function is None and isinstance(tool, dict):
                function = tool.get("function")
            name = getattr(function, "name", None)
            if name is None and isinstance(function, dict):
                name = function.get("name")
            if name:
                names.add(str(name))
        return names

    @staticmethod
    def _candidate_chunks(model_output: str) -> list[str]:
        text = model_output.strip()
        if not text:
            return []
        if "<|assistant|>" in text:
            chunks = re.split(r"<\|assistant\|>", text)
        else:
            chunks = [text]
        return [Glm40414ToolParser._trim_special_tail(chunk) for chunk in chunks]

    @staticmethod
    def _trim_special_tail(text: str) -> str:
        text = text.strip()
        for token in (
            "<|observation|>",
            "<|user|>",
            "<|system|>",
            "<|assistant|>",
            "<|endoftext|>",
        ):
            if token in text:
                text = text.split(token, 1)[0].strip()
        return text

    @staticmethod
    def _parse_chunk(chunk: str, tool_names: set[str]) -> tuple[str, str] | None:
        if not chunk:
            return None
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if len(lines) < 2:
            return None
        name = lines[0]
        if name not in tool_names:
            return None

        raw_args = "\n".join(lines[1:]).strip()
        raw_args = Glm40414ToolParser._strip_code_fence(raw_args)
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw_args)
            except (SyntaxError, ValueError):
                return None
        return name, json.dumps(parsed, ensure_ascii=False)

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text
