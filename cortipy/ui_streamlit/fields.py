"""Form field schema + value coercion for the Streamlit UI.

Extracted from apps/streamlit_app.py (modularization). Depends only on the static
constants module — no Streamlit or app dependencies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from cortipy.ui_streamlit.constants import (
    INT_FIELD_NAMES,
    INT_FIELD_PREFIXES,
    INT_FIELD_SUFFIXES,
)


@dataclass(frozen=True)
class FieldSchema:
    name: str
    kind: str
    options: List[str]
    tooltip: str


def default_values(fields: Iterable[FieldSchema]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for field in fields:
        if field.kind == "dropdown":
            defaults[field.name] = field.options[0] if field.options else ""
        elif field.kind == "numeric":
            defaults[field.name] = None
        else:
            defaults[field.name] = ""
    return defaults


def resolve_choice(options: List[str], current: Optional[str]) -> str:
    if current in options:
        return current
    if current is not None:
        current_str = str(current)
        if current_str in options:
            return current_str
    return options[0] if options else ""


def is_integer_field(name: str) -> bool:
    if name in INT_FIELD_NAMES:
        return True
    lower = name.lower()
    if any(lower.startswith(prefix.lower()) for prefix in INT_FIELD_PREFIXES):
        return True
    if any(lower.endswith(suffix.lower()) for suffix in INT_FIELD_SUFFIXES):
        return True
    return False


def coerce_number(value: Any) -> Optional[float | int]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return int(number) if number.is_integer() else number
    return None


def parse_numeric_list(text: str) -> Optional[List[float | int]]:
    tokens = [tok for tok in re.split(r"[,\s;]+", text.strip()) if tok]
    if len(tokens) <= 1:
        return None
    parsed: List[float | int] = []
    for token in tokens:
        number = coerce_number(token)
        if number is None:
            return None
        parsed.append(number)
    return parsed


def convert_value(field: FieldSchema, value: Any) -> Any:
    if value in ("", None):
        return None
    if field.kind == "numeric":
        number = coerce_number(value)
        if number is None:
            return None
        return int(number) if is_integer_field(field.name) else number
    if field.kind == "dropdown":
        casted = coerce_number(value)
        return casted if casted is not None else value
    if field.kind == "edit":
        text = str(value).strip()
        if not text:
            return None
        numeric_list = parse_numeric_list(text)
        return numeric_list if numeric_list is not None else text
    return value
