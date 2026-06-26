"""Typed probe results -> probe_report.json (feeds Phase-0 gate + per-service caps)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SourceProbe:
    name: str
    reachable: bool
    cold_latency_s: float
    warm_latency_s: float
    throughput_mb_s: float
    rate_limited: bool
    n_rows_sampled: int

    def suggested_concurrency(self) -> int:
        return 2 if self.rate_limited else 16


@dataclass
class ProbeReport:
    internet_ok: bool
    notes: str = ""
    sources: list[SourceProbe] = field(default_factory=list)

    def add(self, sp: SourceProbe) -> None:
        self.sources.append(sp)

    def to_json(self, path: str | Path) -> None:
        payload = {
            "internet_ok": self.internet_ok,
            "notes": self.notes,
            "sources": [asdict(s) for s in self.sources],
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "ProbeReport":
        d = json.loads(Path(path).read_text())
        rep = cls(internet_ok=d["internet_ok"], notes=d.get("notes", ""))
        for s in d["sources"]:
            rep.add(SourceProbe(**s))
        return rep
