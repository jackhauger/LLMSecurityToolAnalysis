"""
compress.py — Unified S-expression + columnar trace serializer.

Converts any nested dict/list structure into a dense text format.
Drops null/empty values and known-useless fields (Gemini thought
signatures, empty metadata dicts) that have no forensic value.
Backend-agnostic: works on LangSmith, Langfuse, and Phoenix traces.

Format rules:
  dict            → (key1 val1 key2 val2 ...)
  list[dict] same keys → columnar block
  list[dict] mixed keys → group by discriminator key, columnar per group
  other list      → (val1 val2 ...)
  scalar          → str value (quoted if contains whitespace or parens)
"""

from __future__ import annotations

# Keys dropped at every nesting level — opaque blobs / zero forensic value
_DROP_KEYS: frozenset[str] = frozenset({
    "__gemini_function_call_thought_signatures__",
    "id",                    # message UUIDs add noise, not signal
    "invalid_tool_calls",    # always empty list in normal runs
    "safety_ratings",        # always empty list
})


def _needs_quoting(s: str) -> bool:
    return any(c in s for c in (' ', '\t', '\n', '(', ')', '[', ']', '|'))


def _fmt_scalar(v) -> str:
    if v is None:
        return "_"
    s = str(v)
    if _needs_quoting(s):
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return s if s else '""'


def _all_same_keys(items: list[dict]) -> bool:
    if not items:
        return True
    keys = set(items[0].keys())
    return all(set(item.keys()) == keys for item in items)


def _find_discriminator(items: list[dict]) -> str | None:
    """Return the key with lowest cardinality present in every item, or None."""
    common = set(items[0].keys())
    for item in items[1:]:
        common &= set(item.keys())
    if not common:
        return None
    best_key = None
    best_ratio = float("inf")
    n = len(items)
    for key in common:
        unique = len({str(item.get(key)) for item in items})
        ratio = unique / n
        if ratio < best_ratio and unique > 1:
            best_ratio = ratio
            best_key = key
    return best_key if best_ratio < 0.8 else None


def _filter_dict(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in _DROP_KEYS and not _is_empty(v)}


def _columnar_block(items: list[dict], keys: list[str], indent: str) -> str:
    header = "[" + " | ".join(keys) + "]"
    rows = []
    for item in items:
        cells = []
        for k in keys:
            v = item.get(k)
            if _is_empty(v):
                cells.append("_")
            elif isinstance(v, (dict, list)):
                cells.append(serialize(v, indent=indent + "  "))
            else:
                cells.append(_fmt_scalar(v))
        rows.append(indent + "  " + " | ".join(cells))
    return indent + header + "\n" + "\n".join(rows)


def _is_empty(v) -> bool:
    """True for values that carry no information."""
    if v is None:
        return True
    if isinstance(v, (dict, list)) and len(v) == 0:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def serialize(obj, indent: str = "") -> str:
    if obj is None:
        return ""

    if isinstance(obj, dict):
        # Filter useless keys and null/empty values
        items = {
            k: v for k, v in obj.items()
            if k not in _DROP_KEYS and not _is_empty(v)
        }
        if not items:
            return "()"
        inner_parts = []
        for k, v in items.items():
            inner_parts.append((_fmt_scalar(k), serialize(v, indent=indent + "  ")))
        flat = " ".join(f"{k} {v}" for k, v in inner_parts)
        if len(flat) > 120:
            lines = [f"{indent}  {k} {v}" for k, v in inner_parts]
            return "(\n" + "\n".join(lines) + f"\n{indent})"
        return f"({flat})"

    if isinstance(obj, list):
        if not obj:
            return "()"

        # List of dicts: try columnar (operate on filtered dicts)
        if all(isinstance(item, dict) for item in obj):
            filtered = [_filter_dict(item) for item in obj]
            filtered = [f for f in filtered if f]  # drop fully-empty dicts
            if not filtered:
                return "()"
            if _all_same_keys(filtered):
                keys = list(filtered[0].keys())
                return _columnar_block(filtered, keys, indent)
            disc = _find_discriminator(filtered)
            if disc is not None:
                groups: dict[str, list[dict]] = {}
                for item in filtered:
                    key = str(item.get(disc, "_"))
                    groups.setdefault(key, []).append(item)
                parts = []
                for group_val, group_items in groups.items():
                    keys = list(dict.fromkeys(k for item in group_items for k in item.keys()))
                    block = _columnar_block(group_items, keys, indent + "  ")
                    parts.append(f"{indent}  ({group_val}\n{block}\n{indent}  )")
                return "(\n" + "\n".join(parts) + f"\n{indent})"
            parts = [serialize(item, indent=indent + "  ") for item in filtered]
            return "(" + " ".join(parts) + ")"

        # List of scalars or mixed
        parts = [serialize(item, indent=indent + "  ") for item in obj]
        joined = " ".join(parts)
        if len(joined) > 120:
            return "(\n" + "\n".join(f"{indent}  {p}" for p in parts) + f"\n{indent})"
        return f"({joined})"

    return _fmt_scalar(obj)
