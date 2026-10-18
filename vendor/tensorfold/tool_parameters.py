"""Schema-aware decoding of Qwen XML parameter text.

Parts adapted from jayleaton/glm53-tensorfold-spark patch 0620 (Apache-2.0): ``const`` schemas, and ``null`` read as
None for a nullable string (its ``typed_value``), here in the shared decoder that every parser and both streamers use.
Its coercions that change what a resent history renders (quoted scalars, Python literals, integral floats) are not.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from typing import Any


def parameter_schemas(tools: Sequence[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Each offered tool's parameter schemas by lowercase name; a spec that is not an object reads as untyped."""

    result = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool["function"] if isinstance(tool.get("function"), dict) else tool
        parameters = function.get("parameters") or function.get("input_schema") or {}
        properties = parameters.get("properties") or {} if isinstance(parameters, dict) else {}
        result[str(function.get("name", "")).lower()] = {
            name: schema for name, schema in properties.items() if isinstance(schema, dict)
        } if isinstance(properties, dict) else {}
    return result


_TYPED = frozenset({"array", "object", "boolean", "integer", "number", "null"})
_KINDS = {bool: "boolean", int: "integer", float: "number", str: "string", type(None): "null", list: "array",
          dict: "object"}


def schema_types(schema: dict[str, Any]) -> frozenset[str]:
    """The JSON types a parameter schema admits: its ``type`` (a name or a list of names), else those of its
    ``anyOf`` / ``oneOf`` members (none if a member is untyped), else those of its ``enum`` values or ``const``."""

    kind = schema.get("type")
    if isinstance(kind, str):
        return frozenset({kind})
    if isinstance(kind, list):
        return frozenset(k for k in kind if isinstance(k, str))
    for key in ("anyOf", "oneOf"):
        members = schema.get(key)
        if isinstance(members, list) and members:
            found = [schema_types(m) if isinstance(m, dict) else frozenset() for m in members]
            return frozenset().union(*found) if all(found) else frozenset()
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return frozenset(_KINDS.get(type(v), "string") for v in enum)
    if "const" in schema:
        return frozenset({_KINDS.get(type(schema["const"]), "string")})
    return frozenset()


def typed_parameter(schema: dict[str, Any]) -> bool:
    """Whether a value is read as JSON: every type the schema admits is a JSON one other than string (a union with
    string, such as ``["string", "null"]``, keeps the text as written)."""

    kinds = schema_types(schema)
    return bool(kinds) and kinds <= _TYPED


def closed_json(text: str) -> str | None:
    """``text`` with the arrays and objects it left open closed, or None if it doesn't only stop short of them."""

    closers, in_string, escaped = [], False, False
    for ch in text:
        if in_string:
            escaped, in_string = (False, True) if escaped else (ch == "\\", ch != '"')
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            closers.append("]" if ch == "[" else "}")
        elif ch in "]}" and (not closers or closers.pop() != ch):
            return None
    return text.rstrip() + "".join(reversed(closers)) if closers and not in_string else None


_PY_WORDS = {"true": True, "false": False, "none": None, "null": None}


def _python_literal(text: str) -> Any:
    """``text`` as a Python literal (``True``, ``['a']``, ``{'k': None}``), tuples as lists; ValueError if not one."""

    word = _PY_WORDS.get(text.strip().lower(), ...)
    if word is not ...:
        return word

    def lists(v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return [lists(x) for x in v]
        if isinstance(v, dict):
            return {k: lists(x) for k, x in v.items()}
        return v

    try:
        return lists(ast.literal_eval(text.strip()))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        raise ValueError(str(exc)) from None


def nullable_text(schema: dict[str, Any]) -> bool:
    """Whether a text value may also be null (``["string", "null"]``, pydantic's ``Optional[str]``): its value
    ``null`` is None, as vLLM's GLM and Qwen parsers read it and as the templates render a None argument."""

    kinds = schema_types(schema)
    return "string" in kinds and "null" in kinds


def decode_parameter(value: str, schema: dict[str, Any], *, python: bool = True) -> Any:
    """A parameter's text read by its schema: a string schema keeps the text (a nullable one reads ``null`` as None,
    which the template renders back as ``null``); a typed one reads JSON of an admitted type, or the value closed
    (#87), and with ``python`` (Qwen's parsers) Python's spelling of one (``True``, ``None``, ``['a']``). Anything
    else stays as written, so a resent history renders the tokens the model wrote: GLM's parsers pass python=False
    (a quoted ``"5"``, Python's ``True`` or ``2.0`` for an integer are not coerced: their JSON would render
    differently)."""

    if not typed_parameter(schema):
        return None if nullable_text(schema) and value == "null" else value
    kinds = schema_types(schema)

    def admitted(parsed: Any) -> bool:
        valid = {
            "array": isinstance(parsed, list),
            "object": isinstance(parsed, dict),
            "boolean": isinstance(parsed, bool),
            "integer": type(parsed) is int,
            "number": type(parsed) in (int, float),
            "null": parsed is None,
        }
        return any(valid[kind] for kind in kinds)

    # a model can end an object or array value one closer short (#87): the value closed is what it meant
    for text in (value, closed_json(value) if kinds & {"array", "object"} else None):
        if text is None:
            continue
        try:
            parsed = json.loads(text)
            json.dumps(parsed, allow_nan=False)
        except (ValueError, TypeError):
            continue
        return parsed if admitted(parsed) else value
    # not JSON: Python's spelling (True, None, single-quoted strings), as vLLM's Qwen parsers also accept
    if not python:
        return value
    try:
        parsed = _python_literal(value)
        json.dumps(parsed, allow_nan=False)
    except (ValueError, TypeError):
        return value
    return parsed if admitted(parsed) else value
