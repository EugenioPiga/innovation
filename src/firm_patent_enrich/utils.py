from __future__ import annotations

import re
from pathlib import Path


_LEGAL_SUFFIXES = {
    "inc",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "llc",
    "plc",
    "sa",
    "ag",
    "nv",
    "group",
    "holdings",
    "holding",
}


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", name.lower())
    tokens = [t for t in cleaned.split() if t and t not in _LEGAL_SUFFIXES]
    return " ".join(tokens)


def compact_name(name: str) -> str:
    return re.sub(r"\s+", "", normalize_name(name))


def to_cik_str(value: str | int) -> str:
    digits = re.sub(r"\D", "", str(value))
    return digits.zfill(10)


def to_int_or_none(value: str | int | float | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def cache_path(cache_dir: Path, stem: str, suffix: str = ".json") -> Path:
    return cache_dir / f"{stem}{suffix}"


def safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return float(num) / float(den)
