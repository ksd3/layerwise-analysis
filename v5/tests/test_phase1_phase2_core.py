from __future__ import annotations

import dataclasses
import os
import importlib.util
from importlib import import_module

import numpy as np
import pytest

from mmu.config import DEFAULT, SOURCES
from mmu.coordination import build_or_attach_manifest, finalize_ready, partition_units
from mmu.healpix import neighbor_pixels, seed_pixel_set
from mmu.io_atomic import atomic_write_bytes, unit_state, write_done_marker
from mmu.matching import adjudicate, match_unit
from mmu.motion import MissingParallaxPolicy, propagate_to_epoch
from mmu.rate_limit import TokenBroker
from mmu.records import read_jsonl, write_jsonl
from mmu.sources.local_fits import crop_or_pad_flux, dedup_highest_snr
from mmu.sources.lsdb_mmu import candidate_dict_from_crossmatch, collection_uri
from mmu.sources.ztf_s3 import lightcurve_to_fixed, s3_hats_uri
from scripts.build_seed_catalogs import assign_ids_and_assert_unique, namespace_ids
from scripts.convert_to_hats import conversion_report, needs_conversion
from scripts.false_match_report import build_report, confusion_radius, gate_false_match, parse_bin_values, pm_scramble, random_offsets
from scripts.finalize_shard import assign_split, enforce_min_instruments
from scripts.finalize_release import summarize_final_shards
from scripts.plan_work_units import enumerate_units
from scripts.run_source_shard import select_unit
from scripts.upload_hf import build_repo_id
from scripts.validate_release import check_min_modalities, check_uniqueness
from scripts.verify_gaia_xmatch import chunk_for_xmatch
from scripts.verify_markers import audit_states


def test_config_sources_and_immutability():
    assert DEFAULT.schema_version == "v5.1"
    assert str(DEFAULT.release_root) == os.environ.get("OMNISKY_RELEASE_ROOT", "release/v5")
    assert SOURCES["hsc"].radius_arcsec <= 1.0
    assert SOURCES["legacy"].kind == "legacy_hdf5"
    assert SOURCES["legacy"].dataset == os.environ.get("OMNISKY_LEGACY_ROOT", "data/legacy_dr10_south_21")
    assert SOURCES["apogee"].org == "hugging-science"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(DEFAULT, "shard_size", 1)


def test_healpix_helpers_nested():
    pixels = seed_pixel_set(np.array([10.0, 10.1]), np.array([2.0, 2.1]), order=4)
    assert pixels and all(isinstance(p, int) for p in pixels)
    pix = next(iter(pixels))
    assert pix in neighbor_pixels(pix, order=4)


def test_adjudicate_and_candidate_grouping():
    d = candidate_dict_from_crossmatch(np.array([0, 0, 1]), np.array([11, 22, 33]),
                                       np.array([0.5, 0.3, 0.2]), np.ones(3))
    res = adjudicate(d, radius_arcsec=1.0)
    assert res[0]["src_index"] == 22
    assert res[0]["match_ambiguous"] is True
    assert res[1]["n_candidates_within_radius"] == 1


def test_collection_uri():
    assert collection_uri("UniverseTBD", "mmu_desi_edr_sv3") == "hf://datasets/UniverseTBD/mmu_desi_edr_sv3"


def test_io_atomic_state_machine(tmp_path):
    p = tmp_path / "unit.bin"
    assert unit_state(p, manifest_hash="m", schema_version="v5.1", code_sha="c") == "pending"
    atomic_write_bytes(b"ok", p)
    assert unit_state(p, manifest_hash="m", schema_version="v5.1", code_sha="c") == "suspicious"
    write_done_marker(p, manifest_hash="m", schema_version="v5.1", code_sha="c")
    assert unit_state(p, manifest_hash="m", schema_version="v5.1", code_sha="c") == "complete"
    assert unit_state(p, manifest_hash="other", schema_version="v5.1", code_sha="c") == "stale"
    p.write_bytes(b"corrupt")
    assert unit_state(p, manifest_hash="m", schema_version="v5.1", code_sha="c") == "corrupt"


def test_coordination_manifest_and_partitions(tmp_path):
    units = [{"source": "desi", "shard": 0}, {"source": "hsc", "shard": 0}]
    h1, created = build_or_attach_manifest(tmp_path, units, inputs_hash="i")
    h2, created2 = build_or_attach_manifest(tmp_path, units, inputs_hash="i")
    assert h1 == h2 and created is True and created2 is False
    assert partition_units(list(range(5)), partition_id=1, num_partitions=2) == [1, 3]
    assert finalize_ready(0, sources=["desi", "hsc"], completed={("desi", 0), ("hsc", 0)})


def test_manifest_rejects_same_inputs_hash_with_different_units(tmp_path):
    build_or_attach_manifest(tmp_path, [{"source": "desi", "shard": 0}], inputs_hash="same")
    with pytest.raises(ValueError, match="units conflict"):
        build_or_attach_manifest(tmp_path, [{"source": "hsc", "shard": 0}], inputs_hash="same")


def test_seed_ids_are_unique_and_namespaced():
    ids = assign_ids_and_assert_unique(np.array([10.0, 200.0]), np.array([2.0, -30.0]), population="star")
    assert len(set(ids.tolist())) == 2
    assert namespace_ids(np.array([1], dtype=np.int64), "star")[0] == "star:1"
    assert namespace_ids(np.array([1], dtype=np.int64), "star")[0] != namespace_ids(np.array([1], dtype=np.int64), "galaxy")[0]
    high_healpix = np.array([2**61], dtype=np.int64)
    assert namespace_ids(high_healpix, "galaxy")[0] != namespace_ids(high_healpix, "star")[0]
    with pytest.raises(ValueError):
        assign_ids_and_assert_unique(np.array([10.0, 10.0]), np.array([2.0, 2.0]))


def test_plan_and_select_units():
    units = enumerate_units(sources=["desi", "hsc"], n_objects=120_000, shard_size=50_000)
    assert len(units) == 6
    assert select_unit(units, task_id=1, partition_id=0, num_partitions=2)["shard"] == 1


def test_finalize_helpers():
    mask = np.array([[True, True, False], [True, False, False]])
    assert enforce_min_instruments(mask, min_instruments=2).tolist() == [True, False]
    split = assign_split(np.array([10.0, 10.0]), np.array([2.0, 2.0]), nside=8, seed=42)
    assert split[0] == split[1]
    assert set(split.tolist()) <= {"train", "val", "test"}


def test_false_match_helpers():
    dra, ddec = random_offsets(n=1000, r_min_arcsec=5.0, r_max_arcsec=30.0, seed=1)
    r = np.hypot(dra, ddec)
    assert (r >= 5.0 - 1e-6).all() and (r <= 30.0 + 1e-6).all()
    gate = gate_false_match({"ok": 0.0005, "bad": 0.02}, threshold=0.001)
    assert gate["passed_bins"] == ["ok"] and gate["low_confidence_bins"] == ["bad"]
    sra, sdec = pm_scramble(np.array([3.0, 4.0]), np.array([4.0, 3.0]), seed=2)
    np.testing.assert_allclose(np.hypot(sra, sdec), np.array([5.0, 5.0]))
    assert confusion_radius(1.0, np.array([0.0, 1000.0]), 10.0, 0.0, 0.1)[1] > 1.0
    assert parse_bin_values(["high_lat=0.0005"]) == {"high_lat": 0.0005}
    assert build_report({"ok": 0.0}, threshold=0.001)["passed"] is True


def test_validation_upload_marker_xmatch_helpers():
    assert audit_states(["complete", "complete"])["passed"] is True
    assert audit_states(["complete", "corrupt"])["bad_states"] == ["corrupt"]
    assert check_uniqueness(np.array([1, 2, 3]))["ok"] is True
    assert check_uniqueness(np.array([1, 1]))["ok"] is False
    assert check_min_modalities(np.array([2, 3]), 2)["ok"] is True
    assert build_repo_id("omnisky-v5") == "UniverseTBD/omnisky-v5"
    assert chunk_for_xmatch(2_000_001, max_rows=2_000_000) == [(0, 2_000_000), (2_000_000, 2_000_001)]


def test_token_broker_accounting():
    broker = TokenBroker(2)
    assert broker.available == 2
    with broker.token():
        assert broker.in_use == 1
        with broker.token():
            assert broker.in_use == 2
        assert broker.in_use == 1
    assert broker.available == 2
    with pytest.raises(RuntimeError):
        broker.release()


def test_finalize_release_summary(tmp_path):
    p = tmp_path / "final" / "population=galaxy" / "shard=000000.jsonl"
    write_jsonl([{"population": "galaxy", "split": "train"}], p)
    summary = summarize_final_shards([p])
    assert summary["n_rows"] == 1
    assert summary["populations"] == {"galaxy": 1}


def test_conversion_and_ztf_helpers():
    assert needs_conversion(SOURCES["desi"]) is False
    assert needs_conversion(SOURCES["sdss_dr16q"]) is True
    report = conversion_report([{"name": "a", "converted": True}, {"name": "b", "converted": False, "reason": "license"}])
    assert report["converted"] == ["a"]
    assert report["flagged_unconvertible"] == [{"name": "b", "reason": "license"}]
    assert s3_hats_uri("dr24") == "s3://ipac-irsa-ztf/ztf/enhanced/dr24/lc/hats"
    lc = lightcurve_to_fixed([1, 2], [10, 11], [0.1, 0.2], max_len=3)
    assert lc["valid"].tolist() == [True, True, False]


def test_local_fits_pure_helpers():
    keep = dedup_highest_snr(np.array(["a", "a", "b"]), np.array([10.0, 20.0, 5.0]))
    assert keep.tolist() == [1, 2]
    np.testing.assert_allclose(crop_or_pad_flux(np.array([1, 2]), 4)[:2], np.array([1, 2]))
    assert np.isnan(crop_or_pad_flux(np.array([1, 2]), 4)[2])


@pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow not installed locally")
def test_schema_when_pyarrow_available():
    pa = import_module("pyarrow")
    from mmu.schemas import SCHEMA_VERSION, final_schema

    schema = final_schema(image_px=2, n_bands=2, spec_len=3)
    assert schema.field("global_object_id").type == pa.int64()
    assert schema.metadata[b"schema_version"] == SCHEMA_VERSION.encode()


def test_motion_propagation_with_parallax_and_missing_policy():
    out = propagate_to_epoch(ra=np.array([100.0]), dec=np.array([0.0]),
                             pmra=np.array([1000.0]), pmdec=np.array([0.0]),
                             parallax_mas=np.array([50.0]), rv_kms=np.array([0.0]),
                             from_epoch_jyear=2016.0, to_epoch_jyear=2006.0)
    assert abs(out["ra"][0] - 100.0) > 1e-5
    missing = propagate_to_epoch(ra=np.array([100.0]), dec=np.array([0.0]),
                                 pmra=np.array([1000.0]), pmdec=np.array([0.0]),
                                 parallax_mas=np.array([np.nan]), rv_kms=np.array([0.0]),
                                 from_epoch_jyear=2016.0, to_epoch_jyear=2006.0,
                                 policy=MissingParallaxPolicy.FLAG)
    assert missing["motion_flag"][0] == "missing_parallax"
    dropped = propagate_to_epoch(ra=np.array([100.0]), dec=np.array([0.0]),
                                 pmra=np.array([1000.0]), pmdec=np.array([0.0]),
                                 parallax_mas=np.array([-1.0]), rv_kms=np.array([0.0]),
                                 from_epoch_jyear=2016.0, to_epoch_jyear=2006.0,
                                 policy=MissingParallaxPolicy.DROP)
    assert bool(dropped["drop"][0]) is True


def test_population_aware_match_unit():
    galaxy = match_unit({"ra": np.array([1.0]), "dec": np.array([2.0])}, {"epoch_jyear": 2010.0}, population="galaxy")
    assert galaxy["ra"].tolist() == [1.0]
