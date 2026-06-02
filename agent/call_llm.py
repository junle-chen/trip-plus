"""OpenAI-compatible chat-completion client helpers.

Model aliases and request defaults are loaded from `models_config.json`.
"""
import json
import os
import random
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import openai
import httpx

MODEL_ALIASES = {
    # Accepted shorthand names used by scripts and local configs.
    "gemini-3-pro-preview": "gemini-3.1-pro-preview",
    "gemini_3_flash": "gemini-3-flash-preview",
}

_ENV_REFERENCE_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _object_extra_content(obj: Any) -> Optional[Any]:
    if isinstance(obj, dict):
        return obj.get("extra_content")
    extra_content = getattr(obj, "extra_content", None)
    if extra_content is not None:
        return extra_content
    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get("extra_content")
    return None


def _message_to_plain_dict(message: Any) -> Dict[str, Any]:
    """Convert SDK message objects to plain OpenAI-compatible dicts."""
    if isinstance(message, dict):
        plain = dict(message)
    else:
        plain = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", "") or "",
        }
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            plain["tool_call_id"] = tool_call_id
        name = getattr(message, "name", None)
        if name:
            plain["name"] = name

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            plain_calls = []
            for tool_call in tool_calls:
                function_obj = getattr(tool_call, "function", None)
                plain_call = {
                    "id": getattr(tool_call, "id", "") or "",
                    "type": "function",
                    "function": {
                        "name": getattr(function_obj, "name", "") if function_obj else "",
                        "arguments": getattr(function_obj, "arguments", "") if function_obj else "",
                    },
                }
                extra_content = _object_extra_content(tool_call)
                if extra_content is not None:
                    plain_call["extra_content"] = extra_content
                plain_calls.append(plain_call)
            if plain_calls:
                plain["tool_calls"] = plain_calls

    plain["role"] = plain.get("role", "assistant")
    plain["content"] = plain.get("content", "") or ""
    return plain


def _message_role_trace(messages: List[Dict[str, Any]]) -> str:
    """Return a compact role trace for debugging invalid chat histories."""
    return " -> ".join(str(message.get("role", "assistant")) for message in messages)


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize messages for OpenAI-compatible chat APIs.

    We always serialize SDK message objects to dicts. For chat-template
    compatibility, we only apply a minimal fix: if there is exactly one system
    message and it is not first, move it to the beginning. Multiple system
    messages are treated as invalid because silently merging them changes prompt
    semantics.
    """
    serialized = [_message_to_plain_dict(message) for message in messages]
    system_indices = [
        index for index, message in enumerate(serialized)
        if message.get("role") == "system"
    ]

    if not system_indices:
        return serialized

    if len(system_indices) == 1:
        system_index = system_indices[0]
        if system_index == 0:
            return serialized
        reordered = list(serialized)
        system_message = reordered.pop(system_index)
        reordered.insert(0, system_message)
        print(
            "  ⚠️  Reordered a single late system message for chat-template compatibility. "
            f"roles: {_message_role_trace(serialized)}"
        )
        return reordered

    raise ValueError(
        "Invalid chat history: multiple system messages found. "
        f"roles: {_message_role_trace(serialized)}"
    )


def load_model_config(model_name: str) -> Dict[str, Any]:
    """
    Load model configuration from models_config.json
    
    Searches for models_config.json in the following order:
    1. Current domain directory (./)
    2. Parent directory (project root)
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration dict
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If model not found in config
    """
    # Try domain directory first
    domain_config_path = Path(__file__).parent.parent / 'models_config.json'
    # Try project root (parent of domain directory)
    root_config_path = Path(__file__).parent.parent.parent / 'models_config.json'
    
    config_path = None
    if domain_config_path.exists():
        config_path = domain_config_path
    elif root_config_path.exists():
        config_path = root_config_path
    else:
        raise FileNotFoundError(
            f"models_config.json not found in:\n"
            f"  - Domain directory: {domain_config_path}\n"
            f"  - Project root: {root_config_path}\n"
            f"Please create models_config.json in the project root or domain directory."
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    models = config.get('models', {})
    resolved_name = MODEL_ALIASES.get(model_name, model_name)
    if resolved_name not in models:
        available = ', '.join(models.keys())
        raise ValueError(
            f"Model '{model_name}' not found in models_config.json\n"
            f"Available models: {available}"
        )
    
    model_config = dict(models[resolved_name])
    base_url = str(model_config.get("base_url", ""))
    match = _ENV_REFERENCE_RE.fullmatch(base_url)
    if match:
        model_config["base_url_env"] = match.group(1)
        model_config["base_url"] = os.getenv(match.group(1), "")
    return model_config


def create_client(model_name: str, model_config: Optional[Dict[str, Any]] = None):
    """
    Create appropriate client based on model configuration
    
    Args:
        model_name: Name of the model
        model_config: Model configuration (if None, will load from config file)
        
    Returns:
        Initialized client instance
    """
    if model_config is None:
        model_config = load_model_config(model_name)
    
    model_type = model_config.get('model_type', 'openai')
    base_url = model_config['base_url']
    api_key_env = model_config.get('api_key_env')
    api_key = os.getenv(api_key_env) if api_key_env else None
    timeout = model_config.get('timeout', 180.0)

    if not base_url:
        base_url_env = model_config.get("base_url_env")
        if base_url_env:
            raise RuntimeError(
                f"Base URL not found for model '{model_name}'\n"
                f"Please set environment variable: {base_url_env}"
            )
        raise RuntimeError(f"Base URL not configured for model '{model_name}'")
    
    if not api_key:
        raise RuntimeError(
            f"API key not found for model '{model_name}'\n"
            f"Please set environment variable: {api_key_env}"
        )
    
    if model_type == 'openai':
        # OpenAI and OpenAI-compatible APIs (Qwen, DeepSeek, etc.)
        # Increased timeout to handle long generation times in planning tasks
        client_kwargs = {"api_key": api_key, "base_url": base_url, "timeout": timeout}
        if '127.0.0.1' in base_url or 'localhost' in base_url:
            client_kwargs["http_client"] = httpx.Client(trust_env=False)
        return openai.OpenAI(**client_kwargs)
    else:
        raise NotImplementedError(
            f"Model type '{model_type}' is not currently supported. "
            f"Supported types: openai"
        )


def call_llm(
    config_name: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    request_overrides: Optional[Dict[str, Any]] = None,
):
    """
    Universal LLM call with automatic client creation and retry logic
    
    Args:
        config_name: Configuration name from models_config.json (display name)
        messages: Message list
        tools: Tool definitions (optional)
    
    Returns:
        API response object
        
    Note:
        All parameters (model_name, temperature, extra_body, etc.) are loaded
        from models_config.json based on the config_name.
    """
    # Load model config and create client
    model_config = load_model_config(config_name)
    client = create_client(config_name, model_config)
    normalized_messages = _normalize_messages(messages)
    
    # Get actual model name for API call (fallback to config_name if not specified)
    actual_model_name = model_config.get('model_name', config_name)
    base_url = model_config.get('base_url', '')
    is_local_endpoint = '127.0.0.1' in base_url or 'localhost' in base_url
    
    # Get parameters from config or use defaults
    temperature = model_config.get('temperature', None)
    top_p = model_config.get('top_p', None)
    seed = model_config.get('seed', None)
    max_tokens = model_config.get('max_tokens', None)
    frequency_penalty = model_config.get('frequency_penalty', None)
    presence_penalty = model_config.get('presence_penalty', None)
    max_retries = model_config.get('max_retries', 6 if is_local_endpoint else 30)
    backoff = model_config.get('backoff', 1.0 if is_local_endpoint else 1.5)
    backoff_max = model_config.get('backoff_max', 8.0 if is_local_endpoint else 30.0)
    jitter_ratio = model_config.get('jitter_ratio', 0.25)
    extra_body = model_config.get('extra_body')  # Get from config
    
    # Detect reasoning models (don't support temperature)
    is_reasoning_model = any(x in actual_model_name.lower() for x in ['o1', 'o3', 'o4-mini', 'reasoner'])
    
    last_err = None
    
    def _extract_retry_after_seconds(err: Exception) -> Optional[float]:
        # OpenAI-compatible SDK errors may include response headers.
        response = getattr(err, "response", None)
        if not response:
            return None
        headers = getattr(response, "headers", None)
        if not headers:
            return None
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return None

    def _is_connection_like_error(err: Exception) -> bool:
        text = f"{type(err).__name__}: {err}".lower()
        markers = (
            "connection error",
            "connecterror",
            "apiconnectionerror",
            "readtimeout",
            "timed out",
            "connection reset",
            "temporarily unavailable",
            "network is unreachable",
            "name or service not known",
            "dns",
        )
        return any(m in text for m in markers)

    def _status_code(err: Exception) -> Optional[int]:
        status_code = getattr(err, "status_code", None)
        if status_code is not None:
            return status_code
        response = getattr(err, "response", None)
        if response is None:
            return None
        return getattr(response, "status_code", None)

    def _is_non_retryable_request_error(err: Exception) -> bool:
        status_code = _status_code(err)
        if status_code is None:
            return False
        return 400 <= status_code < 500 and status_code not in {408, 409, 429}

    for attempt in range(max_retries):
        try:
            params = {
                "model": actual_model_name,
                "messages": normalized_messages,
            }
            
            if tools:
                params["tools"] = tools
            
            if not is_reasoning_model:
                if temperature is not None:
                    params["temperature"] = temperature
                if top_p is not None:
                    params["top_p"] = top_p

            if seed is not None:
                params["seed"] = seed
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            if frequency_penalty is not None:
                params["frequency_penalty"] = frequency_penalty
            if presence_penalty is not None:
                params["presence_penalty"] = presence_penalty
            
            if extra_body:
                params["extra_body"] = extra_body
            if request_overrides:
                params.update(request_overrides)
            response = client.chat.completions.create(**params)
            
            # Validate response
            msg = response.choices[0].message
            has_content = msg.content and msg.content.strip()
            has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
            
            if not has_content and not has_tool_calls:
                raise ValueError("Model returned an empty response without tool calls")
            
            return response
            
        except Exception as e:
            last_err = e

            if _is_non_retryable_request_error(e):
                raise
            
            if attempt == max_retries - 1:
                raise
            
            # Exponential backoff + jitter avoids synchronized retry storms.
            base_wait = min(backoff_max, backoff * (2 ** attempt))
            jitter_span = base_wait * max(0.0, jitter_ratio)
            wait_time = max(0.0, base_wait + random.uniform(-jitter_span, jitter_span))
            retry_after = _extract_retry_after_seconds(e)
            if retry_after is not None:
                wait_time = max(wait_time, retry_after)

            # Recreate client after connection-class failures to recover stale transport state.
            if _is_connection_like_error(e):
                client = create_client(config_name, model_config)

            print(f"  ⚠️  LLM API error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
            print(f"     Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
    
    raise last_err if last_err else RuntimeError("LLM API call failed")
