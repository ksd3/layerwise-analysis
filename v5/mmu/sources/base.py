"""Source adapter protocol."""
from __future__ import annotations

from typing import Protocol


class DataSource(Protocol):
    name: str
    modality: str

    def read_pixel(self, order: int, pixel: int): ...

    def to_rows(self, matched): ...
