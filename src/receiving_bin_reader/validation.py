from __future__ import annotations

import re
from pathlib import Path


_BINISH = re.compile(r"^[A-Z0-9]{2,8}$")


def normalize_bin(value: str) -> str:
    """
    Normalize formatting only.

    No fuzzy correction:
      R 1 B -> R1B
      1a17 -> 1A17
      RIB  -> RIB (still invalid unless explicitly approved)
    """
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def load_approved_bins(path: str | Path) -> set[str]:
    approved: set[str] = set()

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        normalized = normalize_bin(line)
        if normalized:
            approved.add(normalized)

    if not approved:
        raise ValueError("Approved BIN list is empty.")

    return approved


def split_bin_candidates(text: str) -> list[str]:
    """Split obvious multi-BIN text without inventing values."""
    raw_parts = re.split(r"[,;/\n|]+", str(text or ""))
    candidates: list[str] = []

    for part in raw_parts:
        normalized = normalize_bin(part)
        if not normalized:
            continue
        if normalized not in candidates:
            candidates.append(normalized)

    return candidates


def validate_candidates(
    candidates: list[str],
    approved: set[str],
) -> tuple[list[str], list[str]]:
    accepted: list[str] = []
    rejected: list[str] = []

    for candidate in candidates:
        if candidate in approved:
            if candidate not in accepted:
                accepted.append(candidate)
        elif _BINISH.match(candidate):
            if candidate not in rejected:
                rejected.append(candidate)

    return accepted, rejected
