"""Path selection and safe index parsing.

Patterns
--------
* ``"fc1"``                exact path
* ``"layers.*.fc"``        ``*`` matches one path segment, ``**`` any number
* ``"re:blocks\\.[0-3]\\..*"`` a regular expression
* ``"fc1.weight[:, :10]"`` a parameter path followed by a numpy-style index
"""

from __future__ import annotations

import ast
import functools
import re
from typing import Any

import numpy as np


@functools.lru_cache(maxsize=512)
def _glob_to_regex(pattern: str) -> re.Pattern:
    if pattern.startswith("re:"):
        return re.compile(pattern[3:] + r"\Z")
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        if c == "*":
            out.append(r"[^.]*")
        elif c == "?":
            out.append(r"[^.]")
        elif c == "[":
            j = pattern.find("]", i)
            if j == -1:
                out.append(re.escape(c))
            else:
                out.append(pattern[i : j + 1])
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("".join(out) + r"\Z")


def is_pattern(s: str) -> bool:
    return s.startswith("re:") or any(ch in s for ch in "*?[")


def match(pattern: str, path: str) -> bool:
    return bool(_glob_to_regex(pattern).match(path))


def filter_paths(patterns, paths) -> list[str]:
    if isinstance(patterns, str):
        patterns = [patterns]
    out = []
    for p in paths:
        if any(match(pat, p) for pat in patterns):
            out.append(p)
    return out


# ----------------------------------------------------------------- indexing
def split_index(s: str) -> tuple[str, str | None]:
    """``"fc1.weight[:, 3]"`` -> ``("fc1.weight", "[:, 3]")``."""
    s = s.strip()
    if s.endswith("]") and "[" in s and not s.startswith("re:"):
        depth = 0
        for i in range(len(s) - 1, -1, -1):
            if s[i] == "]":
                depth += 1
            elif s[i] == "[":
                depth -= 1
                if depth == 0:
                    head = s[:i]
                    if head and not head.endswith("."):
                        return head, s[i:]
                    break
    return s, None


_ALLOWED = (ast.Expression, ast.Subscript, ast.Name, ast.Load, ast.Constant, ast.Slice, ast.Tuple,
            ast.List, ast.UnaryOp, ast.USub, ast.UAdd)


def parse_index(text: str | None):
    """Safely parse ``"[:, 0:10, ::2]"`` into an index usable on numpy arrays."""
    if text is None:
        return None
    text = text.strip()
    if not text.startswith("["):
        text = f"[{text}]"
    tree = ast.parse("_" + text, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"unsupported index syntax in {text!r}: {type(node).__name__}")
    return _eval_index(tree.body.slice)


def _eval_index(node):
    if isinstance(node, ast.Tuple):
        return tuple(_eval_index(e) for e in node.elts)
    if isinstance(node, ast.Slice):
        f = lambda n: None if n is None else _eval_index(n)  # noqa: E731
        return slice(f(node.lower), f(node.upper), f(node.step))
    if isinstance(node, ast.List):
        return [_eval_index(e) for e in node.elts]
    if isinstance(node, ast.Constant):
        return node.value  # ints, None, Ellipsis
    if isinstance(node, ast.UnaryOp):
        v = _eval_index(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Name):
        if node.id == "None":
            return None
        raise ValueError(f"unsupported name {node.id!r} in index")
    raise ValueError(f"unsupported index node {type(node).__name__}")


def format_index(idx: Any) -> str | None:
    """Inverse of :func:`parse_index` for serializable indices (None if not representable)."""
    if idx is None:
        return None
    if isinstance(idx, str):
        return idx if idx.strip().startswith("[") else f"[{idx}]"
    items = idx if isinstance(idx, tuple) else (idx,)
    parts = []
    for it in items:
        if isinstance(it, slice):
            a = "" if it.start is None else str(it.start)
            b = "" if it.stop is None else str(it.stop)
            parts.append(f"{a}:{b}" + ("" if it.step is None else f":{it.step}"))
        elif it is Ellipsis:
            parts.append("...")
        elif it is None:
            parts.append("None")
        elif isinstance(it, (int, np.integer)):
            parts.append(str(int(it)))
        elif isinstance(it, (list, np.ndarray)):
            arr = np.asarray(it)
            if arr.dtype == bool:
                return None
            parts.append(repr([int(v) for v in arr.ravel()]))
        else:
            return None
    return "[" + ", ".join(parts) + "]"


def normalize_index(idx):
    """Accept strings, ints, slices, tuples, lists, arrays; return a numpy-compatible index."""
    if idx is None:
        return None
    if isinstance(idx, str):
        return parse_index(idx)
    if isinstance(idx, list):
        return np.asarray(idx)
    if isinstance(idx, tuple):
        return tuple(np.asarray(i) if isinstance(i, list) else i for i in idx)
    return idx
