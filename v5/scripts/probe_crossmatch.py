"""Phase-0 proof: one-pixel DESI x HSC LSDB crossmatch + Smith42 concordance."""
from __future__ import annotations
import argparse
import json
from importlib import import_module

import numpy as np
from mmu.concordance import filter_reference_to_healpix_pixel, match_concordance

# NOTE: column suffixes (*_desi/*_hsc) and Smith42 column names must be verified on first cluster run.

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--pixel", type=int, default=257)
    ap.add_argument("--radius-arcsec", type=float, default=1.0)
    ap.add_argument("--tol-arcsec", type=float, default=1.0)
    ap.add_argument("--smith42-revision", required=True,
                    help="immutable Hugging Face dataset revision/commit for Smith42 validation")
    ap.add_argument("--out", default="crossmatch_probe.json")
    args = ap.parse_args()
    lsdb = import_module("lsdb")
    load_dataset = import_module("datasets").load_dataset
    px = lsdb.PixelSearch([(args.order, args.pixel)])

    desi = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_desi_edr_sv3",
                             search_filter=px, columns=["ra", "dec"])
    hsc = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_hsc_pdr3_dud_22.5",
                            search_filter=px, columns=["ra", "dec"])
    matched = desi.crossmatch(hsc, radius_arcsec=args.radius_arcsec, n_neighbors=5).compute()
    our_ra = np.asarray(matched["ra_desi"]); our_dec = np.asarray(matched["dec_desi"])

    ref = load_dataset("Smith42/desi_hsc_crossmatched", split="train", revision=args.smith42_revision)
    ra_col = "ra" if "ra" in ref.column_names else "desi_ra"
    dec_col = "dec" if "dec" in ref.column_names else "desi_dec"
    ref_ra = np.asarray(ref[ra_col]); ref_dec = np.asarray(ref[dec_col])

    ref_ra_px, ref_dec_px = filter_reference_to_healpix_pixel(
        ref_ra, ref_dec, order=args.order, pixel=args.pixel
    )
    conc = match_concordance(our_ra, our_dec, ref_ra_px, ref_dec_px, tol_arcsec=args.tol_arcsec)
    conc["n_matched_pixel"] = int(len(matched))
    conc["n_ref_full"] = int(len(ref_ra))
    conc["n_ref_footprint"] = int(len(ref_ra_px))
    with open(args.out, "w") as f:
        json.dump(conc, f, indent=2, sort_keys=True)
    print(f"matched={len(matched)} recall={conc['recall']:.3f} "
          f"median_sep={conc['median_sep_arcsec']:.3f}\" -> {args.out}")

if __name__ == "__main__":
    main()
