

from typing import Any, Dict


def compact_values(values: Dict[str, Any], max_len: int = 1000) -> Dict[str, Any]:
    """
    Avoid dumping huge payloads into JSONB. Truncates long strings.
    """
    out: Dict[str, Any] = {}
    for k, v in (values or {}).items():
        if isinstance(v, str) and len(v) > max_len:
            out[k] = v[:max_len] + "…"
        else:
            out[k] = v
    return out