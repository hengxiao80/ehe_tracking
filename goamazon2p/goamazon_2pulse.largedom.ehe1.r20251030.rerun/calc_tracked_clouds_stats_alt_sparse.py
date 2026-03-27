#!/usr/bin/env python3
"""
calc_tracked_clouds_stats_alt_sparse.py

Computes per-cloud vertical profiles of area fraction (af), moisture flux (wq),
mass flux (mf), and condensate (qc) for all NT timesteps in parallel.

Key improvements over calc_tracked_clouds_stats_alt.py (claude_opus version):

  1. SPARSE output arrays
     The previous scripts allocated np.zeros((3, NC, NZ)) with NC=190 592, then
     filled only ~8-12% of rows at each timestep.  Per worker:
       af + wq + mf + qc  =  4 × 3 × 190592 × 130 × 8 B  =  2.38 GB (mostly zeros)
     With 128 workers that is 305 GB for output alone, before field arrays.
     This script allocates (3, nc_active, NZ) where nc_active ≤ 16 226, reducing
     output memory to ~0.20 GB per worker.  A 'global_ids' array written alongside
     the data maps each row back to the full cloud index for downstream assembly.

  2. float32 field arrays
     wa_np, qta_np, qn_np, rho_np stored as float32 (SAM output is float32).
     Per worker: 4 × 130 × 1024 × 1024 × 4 B = 2.18 GB (vs 4.36 GB float64).

  Combined memory budget per worker:  ~2.38 GB  (vs ~6.74 GB previously)
  128 workers:  ~305 GB  (vs ~863 GB, which exceeded the 512 GB Perlmutter node
  and caused heavy disk swapping — the dominant bottleneck).

  3. Direct netCDF4 reads
     Bypasses xarray's lazy-loading machinery and the 3-4 full-domain intermediate
     array copies it creates for the shift/mean/subtract pipeline.

  4. Compact HDF5 output (241 files, ~200 MB each vs 2892 pickle files, ~573 GB)
     Each file: global_ids (nc_active,) + {area}_{var} (nc_active, NZ) arrays.

OUTPUT FORMAT (differs from the per-variable pickle format):
  {SCRATCH_HEADER}/pkl/clouds_stats_{ti:08d}.h5
    global_ids  : (nc_active,) int32  — row i → CLOUD_ID_TO_IDX[cloud_id_i]
    core_af     : (nc_active, NZ) float64
    core_wq     : (nc_active, NZ) float64
    core_mf     : (nc_active, NZ) float64
    core_qc     : (nc_active, NZ) float64
    cloud_{af,wq,mf,qc}  ...
    plume_{af,wq,mf,qc}  ...

  To reconstruct full (NC, NT, NZ) arrays in combine_pkl_t, load global_ids and
  scatter:  full_af[global_ids, ti, :] = sparse_af

DEPENDENCIES: h5py, netCDF4, numpy  (no xarray, no numba)
"""

import glob
import json
import os
from multiprocessing import Pool

import h5py
import netCDF4
import numpy as np


# ---------------------------------------------------------------------------
# Configuration — loaded once at module level, shared via fork
# ---------------------------------------------------------------------------
DVS_HEADER    = "/dvs_ro/cfs/cdirs/m1657/xiao169/ehe_tracking/goamazon2p/goamazon_2pulse.largedom.ehe1.r20251030.rerun"
SCRATCH_HEADER = "/pscratch/sd/x/xiao169/ehe_tracking/goamazon2p/goamazon_2pulse.largedom.ehe1.r20251030.rerun"

with open(f"{DVS_HEADER}/uninterrupted_large_clouds.json") as _f:
    UL_CLOUDS_DICT = json.load(_f)
UL_CLOUDS_LIST = list(UL_CLOUDS_DICT.keys())
UL_CLOUDS_SET  = set(UL_CLOUDS_LIST)
NC             = len(UL_CLOUDS_LIST)

# O(1) global cloud-id → row-index lookup (used for global_ids array)
CLOUD_ID_TO_IDX = {int(cid): idx for idx, cid in enumerate(UL_CLOUDS_LIST)}

FILELIST = sorted(glob.glob(f"{DVS_HEADER}/OUT_3D/*.nc"))
SHIFT    = 0
NZ       = 100
NT       = 241

AREA_NAMES    = ["core", "cloud", "plume"]
AREA_DATASETS = ["core", "condensed", "plume"]


# ---------------------------------------------------------------------------
# Field loading — direct netCDF4, returns float32 anomaly arrays
# ---------------------------------------------------------------------------
def _load_fields(ti):
    """
    Read W, QV, QN, density for timestep ti.
    Returns (wa_np, qta_np, qn_np, rho_np), all float32, shape (NZ, NY, NX).

    wa  = cell-centred w − domain-mean w
    qta = (qv + qn) − domain-mean(qv + qn)
    qn  = raw condensate (not anomaly)
    rho = density
    """
    with netCDF4.Dataset(FILELIST[ti + SHIFT]) as ds:
        wi_np = np.asarray(ds["W" ][0], dtype=np.float32)   # staggered-z W
        qv_np = np.asarray(ds["QV"][0], dtype=np.float32)
        qn_np = np.asarray(ds["QN"][0], dtype=np.float32)

    with netCDF4.Dataset(f"{DVS_HEADER}/density/density_{ti:08d}.nc") as ds:
        rho_np = np.asarray(ds["density"][0], dtype=np.float32)

    qt_np = qv_np + qn_np
    del qv_np

    # Interpolate staggered W to cell centres:
    #   w[z] = (wi[z] + wi[z+1]) / 2  for z < NZ-1
    #   w[NZ-1] = wi[NZ-1] / 2        (fill_value=0 at top, matching xarray shift)
    w_np      = wi_np.copy()
    w_np[:-1] = (wi_np[:-1] + wi_np[1:]) * np.float32(0.5)
    w_np[-1] *= np.float32(0.5)
    del wi_np

    # Domain-mean anomalies; keepdims broadcasts across (NY, NX)
    wa_np  = w_np  - w_np .mean(axis=(1, 2), keepdims=True)
    qta_np = qt_np - qt_np.mean(axis=(1, 2), keepdims=True)
    del w_np, qt_np

    return wa_np, qta_np, qn_np, rho_np


# ---------------------------------------------------------------------------
# Per-timestep worker
# ---------------------------------------------------------------------------
def process_t(ti):
    with h5py.File(f"{DVS_HEADER}/hdf5/clouds_{ti:08d}.h5", "r") as clouds_in:
        clouds_list = [k for k in clouds_in.keys() if k in UL_CLOUDS_SET]
        nc_active   = len(clouds_list)

        # Sparse output: rows correspond only to clouds present at this timestep
        af = np.zeros((3, nc_active, NZ))
        wq = np.zeros((3, nc_active, NZ))
        mf = np.zeros((3, nc_active, NZ))
        qc = np.zeros((3, nc_active, NZ))

        if nc_active > 0:
            # Local 0-based index for this timestep's active clouds
            ic_local = {int(cid): i for i, cid in enumerate(clouds_list)}

            wa_np, qta_np, qn_np, rho_np = _load_fields(ti)
            nz_file, ny, nx = wa_np.shape
            if nz_file != NZ:
                raise ValueError(
                    f"NZ mismatch at ti={ti}: configured {NZ}, file has {nz_file}"
                )
            nynx = ny * nx

            for cid in clouds_list:
                ic      = ic_local[int(cid)]
                c_group = clouds_in[cid]

                for i_area, area in enumerate(AREA_DATASETS):
                    c = np.asarray(c_group[area], dtype=np.int64)
                    if c.size == 0:
                        continue

                    # Decompose flat index → (z, y, x)
                    c_z = c // nynx
                    rem = c % nynx
                    c_y = rem // nx
                    c_x = rem % nx

                    # Gather field values at cloud cells (float32)
                    # wa_c cached to avoid reading wa_np twice (wq and mf both need it)
                    wa_c = wa_np[c_z, c_y, c_x]

                    af[i_area, ic, :] = np.bincount(c_z, minlength=NZ)[:NZ]
                    wq[i_area, ic, :] = np.bincount(
                        c_z, weights=wa_c * qta_np[c_z, c_y, c_x], minlength=NZ
                    )[:NZ]
                    mf[i_area, ic, :] = np.bincount(
                        c_z, weights=wa_c * rho_np[c_z, c_y, c_x], minlength=NZ
                    )[:NZ]
                    qc[i_area, ic, :] = np.bincount(
                        c_z, weights=qn_np[c_z, c_y, c_x], minlength=NZ
                    )[:NZ]

    # Map local rows → global cloud indices for downstream assembly
    global_ids = np.fromiter(
        (CLOUD_ID_TO_IDX[int(cid)] for cid in clouds_list),
        dtype=np.int32,
        count=nc_active,
    )

    _write_outputs(ti, global_ids, af, wq, mf, qc)
    print(f"[{ti:3d}/{NT}] {nc_active} clouds", flush=True)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _write_outputs(ti, global_ids, af, wq, mf, qc):
    """Write one compressed HDF5 per timestep (replaces 12 pickle files)."""
    path = f"{SCRATCH_HEADER}/hdf5/clouds_stats_{ti:08d}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("global_ids", data=global_ids, compression="gzip")
        opts = dict(compression="gzip", compression_opts=4)
        for i_area, area_name in enumerate(AREA_NAMES):
            f.create_dataset(f"{area_name}_af", data=af[i_area], **opts)
            f.create_dataset(f"{area_name}_wq", data=wq[i_area], **opts)
            f.create_dataset(f"{area_name}_mf", data=mf[i_area], **opts)
            f.create_dataset(f"{area_name}_qc", data=qc[i_area], **opts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"NC={NC}, NT={NT}, NZ={NZ}")
    print(f"files={len(FILELIST)}")
    # print(f"Peak active clouds per timestep: ~16226 (from prior run)", flush=True)
    # print(f"Estimated peak memory: ~2.4 GB/worker × 128 = ~305 GB (node has 512 GB)", flush=True)

    assert len(FILELIST) >= NT + SHIFT, (
        f"Expected >= {NT + SHIFT} NetCDF files, found {len(FILELIST)}"
    )
    os.makedirs(f"{SCRATCH_HEADER}/hdf5", exist_ok=True)

    with Pool(processes=128) as pool:
        pool.map(process_t, range(NT))
