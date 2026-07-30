"""Versioned Arrow schemas for OmniSky v5 outputs."""
from __future__ import annotations
from importlib import import_module

SCHEMA_VERSION = "v5.1"


def final_schema(*, image_px: int = 160, n_bands: int = 4, spec_len: int = 7781, n_sources: int = 3):
    pa = import_module("pyarrow")

    return pa.schema([
        ("global_object_id", pa.int64()),
        ("object_uid", pa.string()),
        ("seed_ra_deg", pa.float64()),
        ("seed_dec_deg", pa.float64()),
        ("population", pa.string()),
        ("n_instruments_present", pa.int16()),
        ("instrument_presence_mask", pa.list_(pa.bool_(), n_sources)),
        ("split", pa.string()),
        ("hsc_image", pa.list_(pa.float32(), image_px * image_px * n_bands)),
        ("desi_spectrum", pa.list_(pa.float32(), spec_len)),
        ("low_confidence", pa.bool_()),
    ], metadata={b"schema_version": SCHEMA_VERSION.encode()})
