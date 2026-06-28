from __future__ import annotations

import json

import pytest

from scripts.local_lsdb_dry_run import (
    DEFAULT_SMITH42_REVISION,
    PixelSpec,
    SearchSpec,
    build_report,
    import_lsdb_or_exit,
    is_no_coverage_error,
    output_path,
    output_search_path,
    parse_byte_budget,
    parse_catalog_overrides,
    parse_columns,
    parse_cones,
    parse_pixels,
    pixel_searches,
    rows_that_fit,
    select_sources,
)


def test_parse_byte_budget_decimal_and_binary_units():
    assert parse_byte_budget("1GB") == 1_000_000_000
    assert parse_byte_budget("512MiB") == 512 * 1024 * 1024
    assert parse_byte_budget("10_000") == 10_000
    with pytest.raises(ValueError):
        parse_byte_budget("0")
    with pytest.raises(ValueError):
        parse_byte_budget("nope")


def test_parse_pixels_accepts_explicit_or_default_order():
    assert parse_pixels(["4:257"]) == [PixelSpec(order=4, pixel=257)]
    assert parse_pixels(["257", "258"], default_order=4) == [
        PixelSpec(order=4, pixel=257),
        PixelSpec(order=4, pixel=258),
    ]
    with pytest.raises(ValueError, match="bare pixels require"):
        parse_pixels(["257"])
    with pytest.raises(ValueError, match="non-negative"):
        parse_pixels(["4:-1"])


def test_parse_cones_and_pixel_searches():
    assert parse_cones(["150.1,2.2,3600"]) == [
        SearchSpec(kind="cone", label="cone_ra=150.100000_dec=2.200000_radius_arcsec=3600",
                   ra=150.1, dec=2.2, radius_arcsec=3600.0)
    ]
    assert pixel_searches([PixelSpec(order=4, pixel=257)]) == [
        SearchSpec(kind="pixel", label="order=4/pixel=000257", order=4, pixel=257)
    ]
    with pytest.raises(ValueError, match="RA"):
        parse_cones(["360,0,10"])
    with pytest.raises(ValueError, match="positive"):
        parse_cones(["150,2,0"])


def test_columns_sources_and_catalog_overrides():
    assert parse_columns(None) == ["ra", "dec"]
    assert parse_columns("flux") == ["ra", "dec", "flux"]
    assert parse_columns("dec,ra,flux") == ["dec", "ra", "flux"]
    assert [source.name for source in select_sources("desi,hsc")] == ["desi", "hsc"]
    with pytest.raises(ValueError, match="not an LSDB"):
        select_sources("legacy")
    assert parse_catalog_overrides(["desi=/tmp/desi", "hsc=hf://datasets/example/hsc"]) == {
        "desi": "/tmp/desi",
        "hsc": "hf://datasets/example/hsc",
    }


def test_rows_that_fit_and_output_path(tmp_path):
    assert rows_that_fit(n_rows=10, frame_bytes=100, remaining_bytes=200) == 10
    assert rows_that_fit(n_rows=10, frame_bytes=100, remaining_bytes=55) == 5
    assert rows_that_fit(n_rows=0, frame_bytes=100, remaining_bytes=55) == 0
    path = output_path(tmp_path, source="desi", pixel=PixelSpec(order=4, pixel=257), suffix="parquet")
    assert path.as_posix().endswith("source=desi/order=4/pixel=000257.parquet")
    cone_path = output_search_path(
        tmp_path,
        source="hsc",
        search=SearchSpec(kind="cone", label="cone_ra=150.100000_dec=2.200000_radius_arcsec=3600"),
        suffix="jsonl",
    )
    assert cone_path.as_posix().endswith("source=hsc/cone_ra-150.100000_dec-2.200000_radius_arcsec-3600.jsonl")
    assert is_no_coverage_error(ValueError("The selected sky region has no coverage"))


def test_report_shape_and_default_revision():
    report = build_report(
        max_bytes=100,
        stored_bytes=40,
        fetches=[{"source": "desi", "stored_rows": 3}],
        crossmatch={"attempted": True, "matched_rows": 1},
    )
    assert report["remaining_bytes"] == 60
    assert report["cap_reached"] is False
    assert report["crossmatch"]["matched_rows"] == 1
    assert DEFAULT_SMITH42_REVISION == "93d0fddf8c5b61028ee0b6d72fd0dbfa87b38624"
    json.dumps(report)


def test_missing_lsdb_error_is_actionable(monkeypatch: pytest.MonkeyPatch):
    def missing_lsdb(name: str):
        raise ModuleNotFoundError("No module named 'lsdb'", name=name)

    monkeypatch.setattr("scripts.local_lsdb_dry_run.import_module", missing_lsdb)
    with pytest.raises(SystemExit) as excinfo:
        import_lsdb_or_exit()
    message = str(excinfo.value)
    assert "LSDB is not installed" in message
    assert "conda env create -f environment.yml" in message
