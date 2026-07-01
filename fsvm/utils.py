"""Small utilities used across fsvm scripts."""
from __future__ import annotations

import os
from typing import Iterable, List, Optional

import yaml


def _to_dotdict(obj):
    if isinstance(obj, dict):
        d = DotDict(obj)
        for k, v in list(d.items()):
            d[k] = _to_dotdict(v)
        return d
    if isinstance(obj, list):
        return [_to_dotdict(x) for x in obj]
    return obj


def load_config(path: str) -> DotDict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_dotdict(raw)


class DotDict(dict):
    """Recursive attribute access on dicts (for nested yaml)."""

    def __getattr__(self, key):
        try:
            v = dict.__getitem__(self, key)
        except KeyError:
            return None
        if isinstance(v, dict):
            return DotDict(v)
        if isinstance(v, list):
            return [DotDict(x) if isinstance(x, dict) else x for x in v]
        return v

    def __setattr__(self, key, value):
        dict.__setitem__(self, key, value)

    def __delattr__(self, key):
        dict.__delitem__(self, key)


def traverse_dir(
    root_dir: str,
    extension: str = "wav",
    amount: Optional[int] = None,
    str_include: Optional[str] = None,
    str_exclude: Optional[str] = None,
    is_pure: bool = True,
    is_sort: bool = True,
    is_ext: bool = True,
) -> List[str]:
    out: List[str] = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for fn in files:
            if not fn.endswith(extension):
                continue
            mix_path = os.path.join(root, fn)
            pure_path = mix_path[len(root_dir) + 1:] if is_pure else mix_path
            if amount is not None and cnt == amount:
                if is_sort:
                    out.sort()
                return out
            if str_include is not None and str_include not in pure_path:
                continue
            if str_exclude is not None and str_exclude in pure_path:
                continue
            if not is_ext:
                ext = pure_path.split(".")[-1]
                pure_path = pure_path[: -(len(ext) + 1)]
            out.append(pure_path)
            cnt += 1
    if is_sort:
        out.sort()
    return out


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)
