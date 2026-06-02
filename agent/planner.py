"""Tool-using travel planner agent.

The agent calls an OpenAI-compatible chat model, executes registered travel
tools, and returns a wrapped plan, clarification, or no-solution response.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .call_llm import call_llm
from .output_parser import (
    ensure_plan_wrapped,
    extract_plan_content,
)
from .input_processing import (
    load_sample_meta,
)
from .tooling import (
    compact_tool_result_for_context,
    detect_tool_calls,
    duplicate_tool_notice,
    minimize_tool_result_for_context,
    normalize_tool_call,
    tool_call_key,
    known_tool_names,
    TravelToolExecutor,
)


FINAL_ANSWER_PSEUDO_TOOLS = {"plan", "clarification", "no_solution"}


class TravelPlannerAgent:
    """Lightweight function-calling planner over the Trip-Plus tools."""

    def __init__(self,
                 model: str,
                 sample_id: Optional[str] = None,
                 database_base_path: Optional[str] = None,
                 test_data_path: Optional[str] = None,
                 tool_schema_path: Optional[str] = None,
                 language: str = 'en',
                 verbose: bool = False) -> None:
        """
        Initialize Agent
        
        Args:
            model: Model name (must exist in models_config.json)
            sample_id: Sample ID for database path resolution
            database_base_path: Base path to database directory
            tool_schema_path: Path to tool schema JSON file
            language: Language code ('en')
        """
        self._load_env_from_dotenv()
        
        self.model = model
        self.language = language
        
        default_schema = Path(__file__).resolve().parent.parent / 'tools' / f'tool_schema_{language}.json'
        self.tool_schema_path = tool_schema_path or str(default_schema)
        self.verbose = verbose
        self.runtime_stats: Dict[str, Any] = {}
        
        self.sample_id = sample_id
        self.test_data_path = Path(test_data_path) if test_data_path else None
        self.sample_meta = load_sample_meta(self.sample_id, self.test_data_path)
        if database_base_path:
            self.database_base_path = Path(database_base_path)
        else:
            project_root = Path(__file__).resolve().parent.parent
            self.database_base_path = project_root / 'database' / f'database_{language}'

        self.tools_schema = self._load_tool_schemas()
        self.openai_tools = self._build_openai_tools(self.tools_schema)
        self.known_tool_names = set(known_tool_names(self.openai_tools))
        self.tools = TravelToolExecutor(
            sample_id=self.sample_id,
            database_base_path=self.database_base_path,
            test_data_path=self.test_data_path,
            language=self.language,
            sample_meta=self.sample_meta,
        )
        
        if not Path(self.tool_schema_path).exists():
            raise FileNotFoundError(f"Tool schema not found: {self.tool_schema_path}")

    def _load_env_from_dotenv(self) -> None:
        """
        Load environment variables from .env file
        
        Searches for .env in the following order:
        1. Domain directory (./)
        2. Project root (parent of domain)
        """
        domain_root = Path(__file__).resolve().parent.parent
        domain_dotenv = domain_root / '.env'
        project_dotenv = domain_root.parent / '.env'
        dotenv_path = project_dotenv if project_dotenv.exists() else domain_dotenv

        if not dotenv_path.exists():
            return

        for line in dotenv_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

    def _load_tool_schemas(self) -> List[Dict[str, Any]]:
        """Load tool schemas from JSON file"""
        path = Path(self.tool_schema_path)
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict) and 'tools' in raw and isinstance(raw['tools'], list):
            return raw['tools']
        return [raw]

    def _build_openai_tools(self, schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build OpenAI tools format
        - If schema is already {type:function, function:{...}}, use as-is
        - Otherwise wrap as function definition
        """
        tools: List[Dict[str, Any]] = []
        for s in schemas:
            if isinstance(s, dict) and s.get('type') == 'function' and isinstance(s.get('function'), dict):
                tools.append(s)
                continue
            if not isinstance(s, dict):
                continue
            func = {
                "name": s.get('name'),
                "description": s.get('description', ''),
                "parameters": s.get('parameters', {}),
            }
            if func["name"]:
                tools.append({"type": "function", "function": func})
        return tools

    @staticmethod
    def _response_usage_to_dict(response: Any) -> Dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        if isinstance(usage, dict):
            data = dict(usage)
        elif hasattr(usage, "model_dump"):
            data = usage.model_dump()
        elif hasattr(usage, "dict"):
            data = usage.dict()
        else:
            data = {
                key: getattr(usage, key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if getattr(usage, key, None) is not None
            }
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _merge_token_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                total[key] = total.get(key, 0) + value

    @staticmethod
    def _tool_result_has_error(tool_result: str) -> bool:
        try:
            parsed = json.loads(tool_result)
        except Exception:
            return False
        return isinstance(parsed, dict) and bool(parsed.get("error"))

    def _invalid_tool_call_correction(self, tool_name: str) -> str:
        if tool_name.lower() in FINAL_ANSWER_PSEUDO_TOOLS:
            return (
                f"`{tool_name}` is a final-answer tag, not a callable tool. "
                "Do not emit `<plan>`, `<clarification>`, or `<no_solution>` as tool calls; "
                "write one final answer directly as plain text."
            )
        return (
            f"`{tool_name}` is not an available tool. Use only the tools in the provided schema; "
            "if the evidence is already sufficient, stop calling tools and output the final answer."
        )

    def _duplicate_tool_stop_correction(self) -> str:
        return (
            "You have repeatedly called the same tool with the same arguments without new evidence. "
            "Stop calling duplicate tools, reuse the existing tool results, and output the final answer now."
        )

    def _tool_hard_limit_correction(self, hard_limit: int) -> str:
        return (
            f"The tool-call hard limit of {hard_limit} has been reached. Do not call more tools; "
            "reuse the existing tool evidence and output the final answer now."
        )

    @staticmethod
    def _is_runtime_failure_text(text: str) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        return normalized in {
            "reached tool-call hard limit without final answer.",
            "reached max llm calls without final answer.",
        }

    def _set_runtime_stats(
        self,
        *,
        llm_calls: int,
        tool_calls: int,
        tool_executions: int,
        duplicate_tool_calls: int,
        tool_errors: int,
        token_usage: Dict[str, Any],
        llm_call_usage: List[Dict[str, Any]],
        status: str,
    ) -> None:
        self.runtime_stats = {
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "tool_executions": tool_executions,
            "duplicate_tool_calls": duplicate_tool_calls,
            "tool_errors": tool_errors,
            "token_usage": token_usage,
            "llm_call_usage": llm_call_usage,
            "status": status,
        }

    def _assistant_tool_call_message(self, assistant_message, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build an assistant message compatible with following tool results."""
        tool_calls = []
        for call in calls:
            tool_call = {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
            }
            if call.get("extra_content") is not None:
                tool_call["extra_content"] = call["extra_content"]
            tool_calls.append(tool_call)
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls,
        }

    def _message_to_dict(self, msg) -> Dict[str, Any]:
        """Convert message object to serializable dictionary"""
        if isinstance(msg, dict):
            return msg
        
        msg_dict: Dict[str, Any] = {}
        
        # Extract role
        if hasattr(msg, 'role'):
            msg_dict['role'] = msg.role
        elif hasattr(msg, 'get'):
            msg_dict['role'] = msg.get('role', 'assistant')
        else:
            msg_dict['role'] = 'assistant'
        
        # Extract content
        if hasattr(msg, 'content'):
            msg_dict['content'] = msg.content or ''
        elif isinstance(msg, dict) and 'content' in msg:
            msg_dict['content'] = msg['content'] or ''
        else:
            msg_dict['content'] = ''
        
        # Extract tool_calls if present
        tool_calls = getattr(msg, 'tool_calls', None)
        if tool_calls:
            calls_list = []
            for tc in tool_calls:
                try:
                    tool_call_id = getattr(tc, 'id', None) or ''
                    call_dict = {
                        'id': tool_call_id,
                        'type': 'function',
                        'function': {
                            'name': getattr(tc.function, 'name', '') if hasattr(tc, 'function') else '',
                            'arguments': getattr(tc.function, 'arguments', '') if hasattr(tc, 'function') else ''
                        }
                    }
                    extra_content = getattr(tc, 'extra_content', None)
                    if extra_content is None:
                        model_extra = getattr(tc, 'model_extra', None)
                        if isinstance(model_extra, dict):
                            extra_content = model_extra.get('extra_content')
                    if extra_content is not None:
                        call_dict['extra_content'] = extra_content
                    calls_list.append(call_dict)
                except Exception:
                    continue
            if calls_list:
                msg_dict['tool_calls'] = calls_list
        
        # Preserve reasoning_content if present
        if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
            msg_dict['reasoning_content'] = msg.reasoning_content
        
        return msg_dict

    def _serialize_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Convert all messages in list to serializable dictionaries"""
        serialized = []
        for msg in messages:
            serialized.append(self._message_to_dict(msg))
        return serialized

    def run(self,
            user_query: str,
            system_prompt: Optional[str] = None,
            max_llm_calls: int = 100,
            initial_messages: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Agent main loop: Call LLM → Execute tools → Repeat until final answer
        
        Args:
            user_query: User query
            system_prompt: System prompt
            max_llm_calls: Maximum LLM calls
            initial_messages: Optional prior chat messages to prepend before user_query
            
        Returns:
            (final_plan, messages): Final plan and complete message history
        """
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if initial_messages:
            messages.extend(initial_messages)
        messages.append({"role": "user", "content": user_query})
        
        llm_budget = max_llm_calls
        consecutive_tool_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5  # Early termination threshold
        total_tool_calls = 0
        total_tool_executions = 0
        duplicate_tool_calls = 0
        tool_error_count = 0
        llm_call_count = 0
        token_usage: Dict[str, Any] = {}
        llm_call_usage: List[Dict[str, Any]] = []
        # Per agent.run() only. Do not hoist these caches to self/global scope:
        # later user turns must be allowed to re-query the same tool arguments
        # because the active request or itinerary context may have changed.
        tool_result_cache: Dict[str, str] = {}
        duplicate_key_counts: Dict[str, int] = {}
        tool_budget_warning_sent = False
        TOOL_BUDGET_WARNING_THRESHOLD = int(os.getenv("TOOL_BUDGET_WARNING_THRESHOLD", "45"))
        TOOL_CALL_HARD_LIMIT = int(os.getenv("TOOL_CALL_HARD_LIMIT", "64"))
        MAX_DUPLICATE_PER_KEY = int(os.getenv("MAX_DUPLICATE_PER_KEY", "3"))
        # Negative disables the cumulative duplicate limit; per-key duplicate
        # stopping remains controlled by MAX_DUPLICATE_PER_KEY.
        MAX_DUPLICATE_TOOL_CALLS = int(os.getenv("MAX_DUPLICATE_TOOL_CALLS", "-1"))
        USE_CUMULATIVE_DUPLICATE_LIMIT = MAX_DUPLICATE_TOOL_CALLS >= 0
        MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "4000"))
        tool_hard_limit_stop_sent = False
        force_final_without_tools = False
        hit_tool_hard_limit = False
        
        while llm_budget > 0:
            llm_budget -= 1
            llm_call_count += 1
            
            active_tools = None if force_final_without_tools else self.openai_tools
            resp = call_llm(config_name=self.model, messages=messages, tools=active_tools)
            usage = self._response_usage_to_dict(resp)
            if usage:
                llm_call_usage.append({"call_index": llm_call_count, **usage})
                self._merge_token_usage(token_usage, usage)
            
            msg = resp.choices[0].message
            
            if self.verbose:
                prefix = f"[{self.sample_id}] " if self.sample_id else ""
                # Print assistant content/thinking
                content = getattr(msg, 'content', '') or ''
                reasoning = getattr(msg, 'reasoning_content', '') or ''
                if reasoning:
                    print(f"\n{prefix}🤔 Reasoning:\n{reasoning}\n")
                if content:
                    print(f"\n{prefix}🤖 Assistant:\n{content}\n")

            calls = [] if force_final_without_tools else [
                normalize_tool_call(call) for call in detect_tool_calls(msg, self.openai_tools)
            ]
            correction_messages: List[str] = []
            executable_calls: List[Dict[str, Any]] = []
            skipped_duplicate_tool_calls = 0
            skipped_duplicate_key_counts: Dict[str, int] = {}
            if calls:
                projected_tool_calls = total_tool_calls
                projected_seen_tool_keys = set(tool_result_cache)
                projected_duplicate_key_counts = dict(duplicate_key_counts)
                projected_duplicate_tool_calls = duplicate_tool_calls
                for call in calls:
                    tool_name = call["name"]
                    if tool_name not in self.known_tool_names:
                        correction_messages.append(self._invalid_tool_call_correction(tool_name))
                        continue

                    if TOOL_CALL_HARD_LIMIT > 0 and projected_tool_calls >= TOOL_CALL_HARD_LIMIT:
                        total_tool_calls += 1
                        projected_tool_calls += 1
                        hit_tool_hard_limit = True
                        force_final_without_tools = True
                        if not tool_hard_limit_stop_sent:
                            tool_hard_limit_stop_sent = True
                            correction_messages.append(self._tool_hard_limit_correction(TOOL_CALL_HARD_LIMIT))
                        continue

                    tool_key = tool_call_key(tool_name, call["arguments"])
                    is_duplicate_request = tool_key in projected_seen_tool_keys
                    if is_duplicate_request:
                        next_key_duplicates = projected_duplicate_key_counts.get(tool_key, 0) + 1
                        next_duplicate_total = projected_duplicate_tool_calls + 1
                        if (
                            MAX_DUPLICATE_PER_KEY > 0
                            and next_key_duplicates >= MAX_DUPLICATE_PER_KEY
                        ) or (
                            USE_CUMULATIVE_DUPLICATE_LIMIT
                            and next_duplicate_total >= MAX_DUPLICATE_TOOL_CALLS
                        ):
                            projected_duplicate_key_counts[tool_key] = next_key_duplicates
                            projected_duplicate_tool_calls = next_duplicate_total
                            skipped_duplicate_tool_calls += 1
                            skipped_duplicate_key_counts[tool_key] = skipped_duplicate_key_counts.get(tool_key, 0) + 1
                            total_tool_calls += 1
                            projected_tool_calls += 1
                            correction_messages.append(self._duplicate_tool_stop_correction())
                            continue
                        projected_duplicate_key_counts[tool_key] = next_key_duplicates
                        projected_duplicate_tool_calls = next_duplicate_total

                    projected_seen_tool_keys.add(tool_key)
                    projected_tool_calls += 1
                    executable_calls.append(call)

            if executable_calls:
                messages.append(self._assistant_tool_call_message(msg, executable_calls))
            elif calls:
                duplicate_tool_calls += skipped_duplicate_tool_calls
                for tool_key, count in skipped_duplicate_key_counts.items():
                    duplicate_key_counts[tool_key] = duplicate_key_counts.get(tool_key, 0) + count
                # Invalid or excessive tool calls are intentionally not written as assistant
                # tool_calls, because every serialized tool_call would require a matching
                # tool-role response in the next OpenAI-compatible request.
                for correction in dict.fromkeys(correction_messages):
                    messages.append({"role": "user", "content": correction})
                continue
            else:
                messages.append(msg)
            if executable_calls:
                # Execute tool calls and check for errors
                round_has_error = False
                for call in executable_calls:
                    tool_key = tool_call_key(call['name'], call['arguments'])
                    if tool_key in tool_result_cache:
                        duplicate_tool_calls += 1
                        duplicate_key_counts[tool_key] = duplicate_key_counts.get(tool_key, 0) + 1
                        tool_result = duplicate_tool_notice(call['name'])
                        if (
                            MAX_DUPLICATE_PER_KEY > 0
                            and duplicate_key_counts[tool_key] >= MAX_DUPLICATE_PER_KEY
                        ) or (
                            USE_CUMULATIVE_DUPLICATE_LIMIT
                            and duplicate_tool_calls >= MAX_DUPLICATE_TOOL_CALLS
                        ):
                            correction_messages.append(self._duplicate_tool_stop_correction())
                    else:
                        total_tool_executions += 1
                        tool_result = self.tools.call(call['name'], call['arguments'])
                        tool_result_cache[tool_key] = tool_result
                    if self._tool_result_has_error(tool_result):
                        tool_error_count += 1

                    total_tool_calls += 1
                    tool_result_for_context = minimize_tool_result_for_context(
                        call['name'],
                        tool_result,
                    )
                    tool_result_for_context = compact_tool_result_for_context(
                        call['name'],
                        tool_result_for_context,
                        MAX_TOOL_RESULT_CHARS,
                        self.language,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call['id'],
                        "name": call['name'],
                        "content": tool_result_for_context,
                    })

                    if self.verbose:
                        prefix = f"[{self.sample_id}] " if self.sample_id else ""
                        print(f"{prefix}🛠️  Tool Call: {call['name']}({call['arguments']})")
                        # Truncate long results for readability
                        display_res = tool_result_for_context[:500] + "..." if len(tool_result_for_context) > 500 else tool_result_for_context
                        print(f"{prefix}📥 Tool Result: {display_res}\n")

                    # Check if this tool call returned an error
                    if self._tool_result_has_error(tool_result):
                        round_has_error = True
                
                if round_has_error:
                    consecutive_tool_errors += 1
                    if consecutive_tool_errors >= MAX_CONSECUTIVE_ERRORS:
                        error_msg = (f"⛔ Early termination: {consecutive_tool_errors} consecutive rounds "
                                     f"of tool-not-found errors. Tools may not be loaded correctly.")
                        print(error_msg)
                        return f"ERROR: {error_msg}", messages
                else:
                    consecutive_tool_errors = 0  # Reset on success

                if (
                    TOOL_BUDGET_WARNING_THRESHOLD > 0
                    and total_tool_calls >= TOOL_BUDGET_WARNING_THRESHOLD
                    and not tool_budget_warning_sent
                ):
                    tool_budget_warning_sent = True
                    warning = (
                        f"You have already made {total_tool_calls} tool calls. Avoid repeated calls with "
                        "the same tool and arguments, and do not keep querying unrelated attractions, "
                        "origin-city weather, backup hotel brands, or routes not used in the final itinerary. "
                        "Only use new parameters when the minimum required evidence is still missing."
                    )
                    messages.append({"role": "user", "content": warning})

                for correction in dict.fromkeys(correction_messages):
                    messages.append({"role": "user", "content": correction})

                duplicate_tool_calls += skipped_duplicate_tool_calls
                for tool_key, count in skipped_duplicate_key_counts.items():
                    duplicate_key_counts[tool_key] = duplicate_key_counts.get(tool_key, 0) + count
                
                continue
            
            # No tool calls → Return final answer
            raw_final_content = msg.content or ''
            if not raw_final_content.strip():
                self._set_runtime_stats(
                    llm_calls=llm_call_count,
                    tool_calls=total_tool_calls,
                    tool_executions=total_tool_executions,
                    duplicate_tool_calls=duplicate_tool_calls,
                    tool_errors=tool_error_count,
                    token_usage=token_usage,
                    llm_call_usage=llm_call_usage,
                    status="empty_final_after_tool_hard_limit" if hit_tool_hard_limit else "empty_final_answer",
                )
                return "", messages
            if self._is_runtime_failure_text(raw_final_content):
                self._set_runtime_stats(
                    llm_calls=llm_call_count,
                    tool_calls=total_tool_calls,
                    tool_executions=total_tool_executions,
                    duplicate_tool_calls=duplicate_tool_calls,
                    tool_errors=tool_error_count,
                    token_usage=token_usage,
                    llm_call_usage=llm_call_usage,
                    status="runtime_failure_text",
                )
                return "", messages

            final_content = extract_plan_content(raw_final_content)
            final_content = ensure_plan_wrapped(final_content)
            self._set_runtime_stats(
                llm_calls=llm_call_count,
                tool_calls=total_tool_calls,
                tool_executions=total_tool_executions,
                duplicate_tool_calls=duplicate_tool_calls,
                tool_errors=tool_error_count,
                token_usage=token_usage,
                llm_call_usage=llm_call_usage,
                status="completed_after_tool_hard_limit" if hit_tool_hard_limit else "completed",
            )
            print(
                f"   📊 LLM calls: {llm_call_count}, Tool calls: {total_tool_calls} "
                f"(executed: {total_tool_executions}, duplicate: {duplicate_tool_calls})"
            )
            return final_content, messages
        
        print(
            f"   ⚠️ Reached max LLM calls ({max_llm_calls}). LLM calls: {llm_call_count}, "
            f"Tool calls: {total_tool_calls} (executed: {total_tool_executions}, duplicate: {duplicate_tool_calls})"
        )
        self._set_runtime_stats(
            llm_calls=llm_call_count,
            tool_calls=total_tool_calls,
            tool_executions=total_tool_executions,
            duplicate_tool_calls=duplicate_tool_calls,
            tool_errors=tool_error_count,
            token_usage=token_usage,
            llm_call_usage=llm_call_usage,
            status="max_llm_calls",
        )
        return "", messages
