#!/usr/bin/env python3
"""
combine_pkl_t_sparse.py

Assembles the per-timestep sparse HDF5 files produced by
calc_tracked_clouds_stats_alt_sparse.py into full (NC, NT, NZ) combined arrays
and writes them as compressed HDF5 files.

Replaces combine_pkl_t.ipynb for the new sparse output format.

NOTES ON THE EXISTING NOTEBOOK (combine_pkl_t.ipynb)
-----------------------------------------------------
The notebook had two bugs specific to this run directory:
  - scratch path pointed to goamazon_2pulse.largedom.r20251008.rerun (wrong run)
  - nc was probed as 65544 from that wrong path (correct value is NC=190592)
Cells 4-6 in the notebook reference BOMEX paths and are not relevant here.
This script fixes both issues and handles the sparse input format.

INPUT  (from calc_tracked_clouds_stats_alt_sparse.py)
------
  {SCRATCH_HEADER}/hdf5/clouds_stats_{ti:08d}.h5
    global_ids  : (nc_active,) int32
    {area}_{var}: (nc_active, NZ) float64   (area ∈ core/cloud/plume, var ∈ af/wq/mf/qc)

OUTPUT
------
  hdf5/{area}_all_{var}.h5
    '{var}' dataset: (NC, NT, NZ) float32

  To load in downstream notebooks (replace pickle.load calls):
    import h5py, numpy as np
    with h5py.File('hdf5/core_all_af.h5') as f:
        core_af = f['af'][:]   # ndarray shape (NC, NT, NZ)

MEMORY
------
  Peak: NC × NT × NZ × 4 bytes = 190592 × 241 × 130 × 4 ≈ 23.9 GB per array.
  Arrays are processed one at a time (12 total), so ~24 GB peak throughout.
  Suitable for a Perlmutter login node or small interactive job.
"""

import json
import os
import sys

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DVS_HEADER    = "/dvs_ro/cfs/cdirs/m1657/xiao169/ehe_tracking/goamazon2p/goamazon_2pulse.largedom_1024.r20260116"
SCRATCH_HEADER = "/pscratch/sd/x/xiao169/ehe_tracking/goamazon2p/goamazon_2pulse.largedom_1024.r20260116"

# Output directory (relative to cwd, matching existing notebook convention)
OUT_DIR = "hdf5"

NT         = 241
AREA_NAMES = ["core", "cloud", "plume"]
VARS       = ["af", "wq", "mf", "qc"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_cloud_registry():
    """Return NC from uninterrupted_large_clouds.json."""
    with open(f"{DVS_HEADER}/uninterrupted_large_clouds.json") as f:
        d = json.load(f)
    return len(d)


def _probe_nz():
    """Infer NZ from the first available sparse HDF5 timestep file."""
    for ti in range(NT):
        path = f"{SCRATCH_HEADER}/hdf5/clouds_stats_{ti:08d}.h5"
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                return f["core_af"].shape[1]
    raise FileNotFoundError(
        f"No clouds_stats_*.h5 files found under {SCRATCH_HEADER}/hdf5/\n"
        "Run calc_tracked_clouds_stats_alt_sparse.py first."
    )


def _check_completeness():
    """Print a summary of which timestep files are present/missing."""
    present = []
    missing = []
    for ti in range(NT):
        path = f"{SCRATCH_HEADER}/hdf5/clouds_stats_{ti:08d}.h5"
        (present if os.path.exists(path) else missing).append(ti)

    print(f"  Timestep files: {len(present)}/{NT} present", flush=True)
    if missing:
        preview = missing[:10]
        suffix  = f" ... ({len(missing)} total)" if len(missing) > 10 else ""
        print(f"  WARNING: missing timesteps: {preview}{suffix}", flush=True)
        print(f"  Missing timesteps will be left as zeros in the output.", flush=True)
    return missing


# ---------------------------------------------------------------------------
# Core combine routine
# ---------------------------------------------------------------------------
def combine_one(area, var, nc, nz):
    """
    Read all NT sparse HDF5 files for one (area, var) pair and scatter their
    rows into a dense (NC, NT, NZ) float32 array, then write as HDF5.

    Returns the output file size in MB.
    """
    ds_name  = f"{area}_{var}"
    out_path = f"{OUT_DIR}/{area}_all_{var}.h5"

    full = np.zeros((nc, NT, nz), dtype=np.float32)

    for ti in range(NT):
        path = f"{SCRATCH_HEADER}/hdf5/clouds_stats_{ti:08d}.h5"
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            global_ids = f["global_ids"][:]                         # (nc_active,) int32
            sparse     = f[ds_name][:].astype(np.float32)          # (nc_active, NZ)
        full[global_ids, ti, :] = sparse

    os.makedirs(OUT_DIR, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        # shuffle=True rearranges bytes before gzip — substantially improves
        # compression ratio for float arrays (especially sparse ones)
        f.create_dataset(
            var,
            data=full,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            chunks=(min(1000, nc), NT, nz),
        )

    del full
    return os.path.getsize(out_path) / 1e6


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("combine_pkl_t_sparse.py", flush=True)
    print("-" * 50, flush=True)

    NC = _load_cloud_registry()
    NZ = _probe_nz()
    print(f"NC={NC}, NT={NT}, NZ={NZ}", flush=True)
    print(f"Input:  {SCRATCH_HEADER}/hdf5/clouds_stats_*.h5", flush=True)
    print(f"Output: {os.path.abspath(OUT_DIR)}/{{area}}_all_{{var}}.h5", flush=True)
    print(flush=True)

    missing = _check_completeness()
    if missing:
        ans = input(f"\n{len(missing)} timestep(s) missing. Continue with zeros? [y/N] ").strip().lower()
        if ans != "y":
            sys.exit(1)

    mem_gb = NC * NT * NZ * 4 / 1e9
    print(f"\nPeak memory per array: {mem_gb:.1f} GB", flush=True)
    print(f"Processing {len(AREA_NAMES) * len(VARS)} arrays sequentially ...\n", flush=True)

    total_mb = 0.0
    for area in AREA_NAMES:
        for var in VARS:
            label = f"{area}_all_{var}"
            print(f"  {label:20s} ... ", end="", flush=True)
            mb = combine_one(area, var, NC, NZ)
            total_mb += mb
            print(f"{mb:7.0f} MB", flush=True)

    print(f"\nDone.  Total output: {total_mb/1e3:.2f} GB in {OUT_DIR}/", flush=True)
    print(flush=True)
    print("To load arrays in downstream notebooks:", flush=True)
    print("  import h5py, numpy as np", flush=True)
    print("  # replace: pickle.load(open('pkl/core_all_af.pkl','rb'))", flush=True)
    print("  # with:", flush=True)
    print("  with h5py.File('hdf5/core_all_af.h5') as f:", flush=True)
    print("      core_af = f['af'][:]   # shape (NC, NT, NZ)", flush=True)
