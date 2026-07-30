"""Flatiron Legacy DR10 HDF5 cutout reader."""
from __future__ import annotations

from pathlib import Path
from importlib import import_module


def pixel_files(root: str | Path, pixel: int) -> list[Path]:
    return sorted((Path(root) / f"healpix={pixel}").glob("*.hdf5"))


def read_cutouts(path: str | Path):
    h5py = import_module("h5py")

    with h5py.File(path, "r") as f:
        missing = {"ra", "dec", "image_array"} - set(f.keys())
        if missing:
            raise ValueError(f"missing datasets: {sorted(missing)}")
        ra = f["ra"][:]
        dec = f["dec"][:]
        image = f["image_array"][:]
    if len(ra) != len(dec) or len(ra) != len(image):
        raise ValueError("ra/dec/image_array lengths differ")
    return ra, dec, image
