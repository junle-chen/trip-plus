"""Thread-safe wrapper for vLLM's FunctionGemma tool parser.

The built-in parser encodes marker tokens in ``__init__``. vLLM creates parser
instances while handling concurrent requests, and the shared fast tokenizer can
raise ``RuntimeError: Already borrowed`` during that encode path. The inherited
parser logic below only needs the marker strings for our non-streaming use, so
we skip those tokenizer calls.
"""

import regex as re

from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.functiongemma_tool_parser import FunctionGemmaToolParser


@ToolParserManager.register_module("functiongemma_safe")
class SafeFunctionGemmaToolParser(FunctionGemmaToolParser):
    def __init__(self, tokenizer, tools=None):
        ToolParser.__init__(self, tokenizer, tools)

        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []

        self.tool_call_start_token = "<start_function_call>"
        self.tool_call_end_token = "<end_function_call>"
        self.tool_call_start_token_ids = []
        self.tool_call_end_token_ids = []

        self.tool_call_regex = re.compile(
            r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>"
            r"|<start_function_call>call:(\w+)\{(.*)",
            re.DOTALL,
        )
        self.arg_regex = re.compile(
            r"(\w+):<escape>(.*?)<escape>",
            re.DOTALL,
        )
        self.buffered_delta_text = ""
