from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Mapping


CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_\-]*\n([\s\S]*?)\n```", re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return text
    match = CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _json_loads_loose(text: str) -> Dict[str, Any]:
    text = _strip_code_fences(text)
    # 如果文本已经是 JSON 对象，直接解析
    if text.strip().startswith('{') and text.strip().endswith('}'):
        return json.loads(text)
    # 否则尝试解析
    return json.loads(text)


def _case_insensitive_lookup(data: Mapping[str, Any], key: str) -> Any:
    for k, v in data.items():
        if isinstance(k, str) and k.lower() == key.lower():
            return v
    raise KeyError(key)


def _normalize_keys(data: Mapping[str, Any], expected_keys: Iterable[str], aliases: Mapping[str, Iterable[str]] | None = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in expected_keys:
        value = None
        found = False

        # Exact
        if key in data:
            value = data[key]
            found = True
        # Case-insensitive
        if not found:
            try:
                value = _case_insensitive_lookup(data, key)
                found = True
            except KeyError:
                pass
        # Aliases
        if not found and aliases and key in aliases:
            for alias in aliases[key]:
                try:
                    value = _case_insensitive_lookup(data, alias)
                    found = True
                    break
                except KeyError:
                    continue

        result[key] = value
    return result


def parse_llm_response(response: Any, expected_keys: Iterable[str], aliases: Mapping[str, Iterable[str]] | None = None) -> Dict[str, Any]:
    """
    Robustly parse LLM outputs that may be:
    - Direct JSON object (new format)
    - JSON string with/without code fences
    - Dict with None key mapping to a JSON string inside code fences
    - Dict with extraneous keys or wrong casing for expected keys
    Returns a dict containing only expected_keys (values may be None if absent).
    """
    # Case 1: already a mapping containing expected keys
    if isinstance(response, dict):
        # If it contains a None key whose value looks like code-fenced JSON, try to parse that
        if None in response and isinstance(response[None], str):
            try:
                parsed = _json_loads_loose(response[None])
                response = parsed
            except Exception:
                # Fall through to normalization on current mapping
                pass
        else:
            # Some providers wrap the JSON string under arbitrary key; try to find a value that is a fenced JSON
            for v in response.values():
                if isinstance(v, str) and CODE_FENCE_RE.search(v):
                    try:
                        parsed = _json_loads_loose(v)
                        response = parsed
                        break
                    except Exception:
                        continue

        if isinstance(response, dict):
            return _normalize_keys(response, expected_keys, aliases)

    # Case 2: string (possibly fenced)
    if isinstance(response, str):
        parsed = _json_loads_loose(response)
        return _normalize_keys(parsed, expected_keys, aliases)

    # Fallback: empty dict with expected keys set to None
    return {k: None for k in expected_keys}


def coerce_types_information_gathering(data: Dict[str, Any]) -> Dict[str, Any]:
    # floor_count, character_health as int or None
    for k in ("floor_count", "character_health"):
        v = data.get(k)
        if isinstance(v, str):
            if v.lower() == "null":
                data[k] = None
            elif v.isdigit():
                data[k] = int(v)
    # task_horizon as "0"/"1"
    v = data.get("task_horizon")
    if isinstance(v, bool):
        data["task_horizon"] = "1" if v else "0"
    elif v is None:
        data["task_horizon"] = "0"
    return data


def coerce_types_self_reflection(data: Dict[str, Any]) -> Dict[str, Any]:
    v = data.get("success")
    if isinstance(v, str):
        if v.lower() in ("true", "yes"): data["success"] = True
        elif v.lower() in ("false", "no"): data["success"] = False
    return data


def coerce_types_action_planning(data: Dict[str, Any]) -> Dict[str, Any]:
    actions = data.get("actions")
    if isinstance(actions, str):
        data["actions"] = [actions]
    return data


def coerce_types_task_inference(data: Dict[str, Any]) -> Dict[str, Any]:
    return data


