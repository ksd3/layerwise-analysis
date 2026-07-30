"""Bounded local LSDB smoke test for HATS catalog access.

This script is intentionally a dry run: it proves that local/HF LSDB catalogs can
be opened, spatially filtered, materialized, and stored without requiring Delta.
It caps bytes written to disk; callers should still keep pixels and columns small
because LSDB may transiently materialize more than the final stored slice.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable

from mmu.config import SOURCES, Source
from mmu.io_atomic import atomic_write_bytes
from mmu.sources.lsdb_mmu import collection_uri


DEFAULT_SMITH42_REVISION = "93d0fddf8c5b61028ee0b6d72fd0dbfa87b38624"
DEFAULT_DRY_RUN_COLUMNS = ("ra", "dec")


@dataclass(frozen=True, slots=True)
class PixelSpec:
    order: int
    pixel: int


@dataclass(frozen=True, slots=True)
class SearchSpec:
    kind: str
    label: str
    order: int | None = None
    pixel: int | None = None
    ra: float | None = None
    dec: float | None = None
    radius_arcsec: float | None = None


def parse_byte_budget(value: str) -> int:
    """Parse byte budgets like ``1000000000``, ``1GB``, or ``512MiB``."""
    raw = value.strip().lower().replace("_", "")
    if not raw:
        raise ValueError("byte budget cannot be empty")
    suffixes = {
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "b": 1,
    }
    for suffix, multiplier in suffixes.items():
        if raw.endswith(suffix):
            number = raw[: -len(suffix)]
            break
    else:
        number = raw
        multiplier = 1
    try:
        budget = int(float(number) * multiplier)
    except ValueError as exc:
        raise ValueError(f"invalid byte budget: {value!r}") from exc
    if budget <= 0:
        raise ValueError("byte budget must be positive")
    return budget


def parse_pixels(values: Iterable[str], *, default_order: int | None = None) -> list[PixelSpec]:
    """Parse pixel specs from ``ORDER:PIXEL`` strings or bare pixels."""
    pixels: list[PixelSpec] = []
    for value in values:
        raw = value.strip()
        if not raw:
            continue
        if ":" in raw:
            order_s, pixel_s = raw.split(":", 1)
            order = int(order_s)
            pixel = int(pixel_s)
        else:
            if default_order is None:
                raise ValueError("bare pixels require --order")
            order = default_order
            pixel = int(raw)
        if order < 0 or pixel < 0:
            raise ValueError("order and pixel must be non-negative")
        pixels.append(PixelSpec(order=order, pixel=pixel))
    if not pixels:
        raise ValueError("at least one pixel is required")
    return pixels


def parse_cones(values: Iterable[str]) -> list[SearchSpec]:
    """Parse cone specs from ``RA,DEC,RADIUS_ARCSEC`` strings."""
    cones: list[SearchSpec] = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 3 or any(not part for part in parts):
            raise ValueError("cone specs must use RA,DEC,RADIUS_ARCSEC")
        ra, dec, radius_arcsec = (float(part) for part in parts)
        if not (0.0 <= ra < 360.0):
            raise ValueError("cone RA must be in [0, 360)")
        if not (-90.0 <= dec <= 90.0):
            raise ValueError("cone Dec must be in [-90, 90]")
        if radius_arcsec <= 0:
            raise ValueError("cone radius must be positive")
        label = f"cone_ra={ra:.6f}_dec={dec:.6f}_radius_arcsec={radius_arcsec:g}"
        cones.append(SearchSpec(kind="cone", label=label, ra=ra, dec=dec, radius_arcsec=radius_arcsec))
    if not cones:
        raise ValueError("at least one cone is required")
    return cones


def pixel_searches(pixels: Iterable[PixelSpec]) -> list[SearchSpec]:
    return [SearchSpec(kind="pixel", label=f"order={pixel.order}/pixel={pixel.pixel:06d}",
                       order=pixel.order, pixel=pixel.pixel)
            for pixel in pixels]


def parse_columns(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return list(DEFAULT_DRY_RUN_COLUMNS)
    columns = [part.strip() for part in value.split(",") if part.strip()]
    if not columns:
        raise ValueError("at least one column is required")
    if "ra" not in columns or "dec" not in columns:
        columns = ["ra", "dec", *[col for col in columns if col not in {"ra", "dec"}]]
    return columns


def select_sources(names: str) -> list[Source]:
    selected: list[Source] = []
    for name in [part.strip() for part in names.split(",") if part.strip()]:
        if name not in SOURCES:
            raise ValueError(f"unknown source: {name}")
        source = SOURCES[name]
        if source.kind != "lsdb_mmu":
            raise ValueError(f"source {name!r} is {source.kind!r}, not an LSDB MMU HATS source")
        selected.append(source)
    if not selected:
        raise ValueError("at least one source is required")
    return selected


def source_catalog_uri(source: Source, overrides: dict[str, str]) -> str:
    return overrides.get(source.name, collection_uri(source.org, source.dataset))


def import_lsdb_or_exit() -> Any:
    try:
        return import_module("lsdb")
    except ModuleNotFoundError as exc:
        if exc.name != "lsdb":
            raise
        raise SystemExit(
            "LSDB is not installed in this Python environment.\n"
            "Create the project conda env with `conda env create -f environment.yml` "
            "or install it into the active env with `python -m pip install lsdb`.\n"
            "Then rerun with that env's Python, for example: "
            "`python -m scripts.local_lsdb_dry_run --out-dir /tmp/omnisky-lsdb-dry-run`."
        ) from exc


def parse_catalog_overrides(values: Iterable[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("catalog overrides must use SOURCE=URI")
        name, uri = value.split("=", 1)
        name = name.strip()
        uri = uri.strip()
        if not name or not uri:
            raise ValueError("catalog override source and URI must be non-empty")
        overrides[name] = uri
    return overrides


def output_path(out_dir: str | Path, *, source: str, pixel: PixelSpec, suffix: str) -> Path:
    return Path(out_dir) / f"source={source}" / f"order={pixel.order}" / f"pixel={pixel.pixel:06d}.{suffix}"


def output_search_path(out_dir: str | Path, *, source: str, search: SearchSpec, suffix: str) -> Path:
    if search.kind == "pixel":
        if search.order is None or search.pixel is None:
            raise ValueError("pixel search missing order/pixel")
        return output_path(out_dir, source=source, pixel=PixelSpec(search.order, search.pixel), suffix=suffix)
    safe_label = search.label.replace("/", "_").replace("=", "-").replace(",", "_")
    return Path(out_dir) / f"source={source}" / f"{safe_label}.{suffix}"


def dataframe_bytes(df: Any) -> int:
    usage = df.memory_usage(index=True, deep=True)
    return int(usage.sum() if hasattr(usage, "sum") else usage)


def rows_that_fit(*, n_rows: int, frame_bytes: int, remaining_bytes: int) -> int:
    if n_rows <= 0 or frame_bytes <= 0 or remaining_bytes <= 0:
        return 0
    if frame_bytes <= remaining_bytes:
        return n_rows
    approx_bytes_per_row = max(frame_bytes / n_rows, 1.0)
    return max(0, min(n_rows, int(remaining_bytes / approx_bytes_per_row)))


def dataframe_records(df: Any) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    return [dict(row) for row in records]


def write_dataframe(df: Any, path: Path, *, prefer_parquet: bool) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if prefer_parquet:
        try:
            import_module("pyarrow")
        except ModuleNotFoundError:
            prefer_parquet = False
    if prefer_parquet:
        parquet_path = path.with_suffix(".parquet")
        tmp = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
        df.to_parquet(tmp)
        tmp.replace(parquet_path)
        return str(parquet_path), parquet_path.stat().st_size
    jsonl_path = path.with_suffix(".jsonl")
    payload = "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in dataframe_records(df))
    atomic_write_bytes(payload.encode(), jsonl_path)
    return str(jsonl_path), jsonl_path.stat().st_size


def is_no_coverage_error(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and "no coverage" in str(exc).lower()


def make_search_filter(*, lsdb: Any, search: SearchSpec) -> Any:
    if search.kind == "pixel":
        if search.order is None or search.pixel is None:
            raise ValueError("pixel search missing order/pixel")
        return lsdb.PixelSearch([(search.order, search.pixel)])
    if search.kind == "cone":
        if search.ra is None or search.dec is None or search.radius_arcsec is None:
            raise ValueError("cone search missing ra/dec/radius")
        return lsdb.ConeSearch(ra=search.ra, dec=search.dec, radius_arcsec=search.radius_arcsec)
    raise ValueError(f"unknown search kind: {search.kind}")


def fetch_search_dataframe(*, lsdb: Any, uri: str, search: SearchSpec, columns: list[str]) -> Any:
    search_filter = make_search_filter(lsdb=lsdb, search=search)
    catalog = lsdb.open_catalog(uri, search_filter=search_filter, columns=columns)
    return catalog.compute()


def run_crossmatch_smoke(*, lsdb: Any, seed_df: Any, target_uri: str, columns: list[str], radius_arcsec: float, max_seed_rows: int, search: SearchSpec | None) -> dict[str, Any]:
    if len(seed_df) == 0:
        return {"attempted": False, "reason": "no seed rows"}
    seed = seed_df[["ra", "dec"]].head(max_seed_rows).copy()
    seed["dry_seed_id"] = range(len(seed))
    seed_catalog = lsdb.from_dataframe(seed, ra_column="ra", dec_column="dec")
    try:
        if search is None:
            target = lsdb.open_catalog(target_uri, columns=columns)
        else:
            target = lsdb.open_catalog(target_uri, columns=columns, search_filter=make_search_filter(lsdb=lsdb, search=search))
        matched = seed_catalog.crossmatch(target, radius_arcsec=radius_arcsec).compute()
    except ValueError as exc:
        if is_no_coverage_error(exc):
            return {"attempted": False, "reason": "target search region has no coverage"}
        raise
    return {"attempted": True, "seed_rows": int(len(seed)), "matched_rows": int(len(matched))}


def build_report(*, max_bytes: int, stored_bytes: int, fetches: list[dict[str, Any]], crossmatch: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "max_bytes": int(max_bytes),
        "stored_bytes": int(stored_bytes),
        "remaining_bytes": int(max_bytes - stored_bytes),
        "cap_reached": stored_bytes >= max_bytes,
        "fetches": fetches,
        "crossmatch": crossmatch or {"attempted": False},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Bounded local LSDB HATS dry run")
    ap.add_argument("--sources", default="desi,hsc", help="comma-separated LSDB source names from mmu.config")
    ap.add_argument("--order", type=int, default=4, help="default order for bare --pixels values")
    ap.add_argument("--pixels", nargs="+", default=["257"], help="pixel specs as PIXEL or ORDER:PIXEL")
    ap.add_argument("--cone", action="append", default=[], help="use cone search RA,DEC,RADIUS_ARCSEC; can be repeated and overrides --pixels")
    ap.add_argument("--columns", default=None, help="comma-separated columns; ra,dec are added if omitted")
    ap.add_argument("--max-bytes", default="1GB", help="maximum bytes to store locally, e.g. 1GB or 512MiB")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--catalog", action="append", default=[], help="override catalog URI as SOURCE=URI")
    ap.add_argument("--jsonl", action="store_true", help="write JSONL instead of Parquet")
    ap.add_argument("--crossmatch", action="store_true", help="run a tiny first-source to second-source LSDB crossmatch smoke")
    ap.add_argument("--crossmatch-radius-arcsec", type=float, default=1.0)
    ap.add_argument("--max-crossmatch-seed-rows", type=int, default=100)
    ap.add_argument("--smith42-revision", default=DEFAULT_SMITH42_REVISION,
                    help="document the Smith42 revision paired with this dry run")
    args = ap.parse_args()

    max_bytes = parse_byte_budget(args.max_bytes)
    searches = parse_cones(args.cone) if args.cone else pixel_searches(parse_pixels(args.pixels, default_order=args.order))
    columns = parse_columns(args.columns)
    sources = select_sources(args.sources)
    overrides = parse_catalog_overrides(args.catalog)
    lsdb = import_lsdb_or_exit()

    stored_bytes = 0
    fetches: list[dict[str, Any]] = []
    first_frame: Any | None = None
    first_search: SearchSpec | None = None
    first_uri: str | None = None
    second_uri: str | None = None
    stop = False
    for source in sources:
        uri = source_catalog_uri(source, overrides)
        if first_uri is None:
            first_uri = uri
        elif second_uri is None:
            second_uri = uri
        for search in searches:
            if stop:
                break
            try:
                df = fetch_search_dataframe(lsdb=lsdb, uri=uri, search=search, columns=columns)
            except ValueError as exc:
                if not is_no_coverage_error(exc):
                    raise
                fetches.append({
                    "source": source.name,
                    "uri": uri,
                    "search": search.label,
                    "search_kind": search.kind,
                    "order": search.order,
                    "pixel": search.pixel,
                    "columns": columns,
                    "fetched_rows": 0,
                    "fetched_memory_bytes": 0,
                    "stored_rows": 0,
                    "stored_bytes": 0,
                    "output": None,
                    "truncated_to_fit_budget": False,
                    "error": "no_coverage",
                })
                continue
            frame_bytes = dataframe_bytes(df)
            fit_rows = rows_that_fit(n_rows=len(df), frame_bytes=frame_bytes, remaining_bytes=max_bytes - stored_bytes)
            truncated = fit_rows < len(df)
            out_file = None
            written_bytes = 0
            if fit_rows > 0:
                to_write = df.head(fit_rows).copy()
                if first_frame is None:
                    first_frame = to_write
                    first_search = search
                base = output_search_path(args.out_dir, source=source.name, search=search, suffix="parquet")
                out_file, written_bytes = write_dataframe(to_write, base, prefer_parquet=not args.jsonl)
                stored_bytes += written_bytes
            fetches.append({
                "source": source.name,
                "uri": uri,
                "search": search.label,
                "search_kind": search.kind,
                "order": search.order,
                "pixel": search.pixel,
                "columns": columns,
                "fetched_rows": int(len(df)),
                "fetched_memory_bytes": int(frame_bytes),
                "stored_rows": int(fit_rows),
                "stored_bytes": int(written_bytes),
                "output": out_file,
                "truncated_to_fit_budget": truncated,
            })
            stop = stored_bytes >= max_bytes or (truncated and fit_rows == 0)
        if stop:
            break

    crossmatch = {"attempted": False}
    if args.crossmatch:
        if first_frame is None or second_uri is None:
            crossmatch = {"attempted": False, "reason": "need stored rows and at least two sources"}
        else:
            crossmatch = run_crossmatch_smoke(
                lsdb=lsdb,
                seed_df=first_frame,
                target_uri=second_uri,
                columns=columns,
                radius_arcsec=args.crossmatch_radius_arcsec,
                max_seed_rows=args.max_crossmatch_seed_rows,
                search=first_search,
            )

    report = build_report(max_bytes=max_bytes, stored_bytes=stored_bytes, fetches=fetches, crossmatch=crossmatch)
    report["smith42_revision"] = args.smith42_revision
    report_path = Path(args.out_dir) / "local_lsdb_dry_run_report.json"
    atomic_write_bytes(json.dumps(report, indent=2, sort_keys=True).encode(), report_path)
    print(f"stored={stored_bytes} max={max_bytes} report={report_path}")


if __name__ == "__main__":
    main()
