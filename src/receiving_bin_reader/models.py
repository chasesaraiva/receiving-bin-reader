from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass(frozen=True)
class RowGeometry:
    row_number: int
    core: Box
    bin_padded: Box
    core_top_in_bin_crop: int
    core_bottom_in_bin_crop: int


@dataclass(frozen=True)
class PageGeometry:
    page_width: int
    page_height: int
    header_bottom: int
    sku_right: int
    bin_right: int
    rows: list[RowGeometry]


@dataclass(frozen=True)
class OcrToken:
    text: str
    score: float
    box: Box


@dataclass
class RowRead:
    row_number: int
    accepted_bins: list[str] = field(default_factory=list)
    raw_candidates: list[str] = field(default_factory=list)
    rejected_candidates: list[str] = field(default_factory=list)
    mean_score: float = 0.0
    status: str = "REVIEW"
    row_crop: Path | None = None
    bin_crop: Path | None = None
