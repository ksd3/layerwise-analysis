"""HTTP download utilities — proven functions from v2."""

import os
import re
import time
import requests
import numpy as np
import h5py
from html.parser import HTMLParser


class LinkParser(HTMLParser):
    """Parse HTML directory listings for file/directory links."""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for n, v in attrs:
                if n == "href" and not v.startswith(("?", "/", ".", "http")):
                    self.links.append(v)


def list_directory(url):
    """List files/dirs at an HTTP directory URL."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    p = LinkParser()
    p.feed(resp.text)
    return p.links


def list_healpix_dirs(url):
    """List healpix=N directories at a URL. Returns {hp_int: (full_url, dir_name)}."""
    entries = list_directory(url)
    result = {}
    for entry in entries:
        m = re.match(r"healpix=(\d+)/", entry)
        if m:
            result[int(m.group(1))] = (url + entry, entry.rstrip("/"))
    return result


def download_file(url, local_path, max_retries=3):
    """Download a file with retry and resume support."""
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    # Check if already downloaded (same size)
    if os.path.exists(local_path):
        try:
            head = requests.head(url, timeout=10)
            remote_size = int(head.headers.get("Content-Length", 0))
            if remote_size > 0 and os.path.getsize(local_path) == remote_size:
                return local_path, 0
        except Exception:
            pass

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            total = 0
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    total += len(chunk)
            return local_path, total
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def find_hdf5_files(url, depth=0):
    """Recursively find HDF5 files at an HTTP directory URL."""
    if depth > 5:
        return []
    files = []
    entries = list_directory(url)
    for entry in entries:
        if entry.endswith("/"):
            files.extend(find_hdf5_files(url + entry, depth + 1))
        elif entry.endswith((".hdf5", ".h5", ".hdf")):
            files.append(url + entry)
    return files


def download_healpix_cell(dataset_url, subdir, hp, output_dir, dir_name=None):
    """Download all HDF5 files in a healpix cell directory."""
    if dir_name is None:
        dir_name = f"healpix={hp}"
    cell_url = f"{dataset_url}{subdir}/{dir_name}/"
    local_dir = os.path.join(output_dir, subdir, f"healpix={hp}")
    try:
        files = find_hdf5_files(cell_url, depth=0)
        total = 0
        local_files = []
        for furl in files:
            fname = furl.split("/")[-1]
            lpath = os.path.join(local_dir, fname)
            _, nb = download_file(furl, lpath)
            total += nb
            local_files.append(lpath)
        return local_files, total
    except Exception:
        return [], 0


def read_hdf5_data(path, skip_cols=None, columns_to_keep=None):
    """Read HDF5 file, normalize keys to lowercase.

    Args:
        path: Path to HDF5 file
        skip_cols: Set of lowercase column names to skip
        columns_to_keep: If set, only keep these columns (plus ra/dec).
                         None means keep all.
    """
    data = {}
    try:
        with h5py.File(path, "r") as f:
            for key in f.keys():
                lk = key.lower()
                if skip_cols and lk in skip_cols:
                    continue
                # Column filtering: always keep ra/dec, filter the rest
                if columns_to_keep is not None:
                    if lk not in columns_to_keep and lk not in ("ra", "dec"):
                        continue
                dset = f[key]
                if len(dset.shape) > 4 or dset.nbytes > 500 * 1024 * 1024:
                    continue
                val = dset[:]
                if val.dtype.kind in ("S", "O", "U"):
                    val = np.array(val, dtype=str)
                data[lk] = val
    except Exception as e:
        print(f"    WARN: Failed to read {path}: {e}")
    return data
