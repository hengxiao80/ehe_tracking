#!/usr/bin/env python3
"""
calc_csd_sparse.large_1024.py

Computes cloud-size distribution statistics for the largedom_1024 runs
(EHE1 r20260115 and CTL r20260116) and saves results as pickle files.

Replaces the computation cells in calc_csd_sparse.large_1024.ipynb.

OUTPUT
------
  {run}/pkl/csd_stats.sparse.claude.pkl   — tuple of 10 arrays (see calculate_csd)
"""

import json
import pickle
from pathlib import Path

import h5py
import numpy as np


def calculate_csd(sim_path, scratch_path=None):
    """Calculates cloud statistics for a given simulation.

    Parameters
    ----------
    sim_path    : path containing uninterrupted_large_clouds.json and pkl/qclm.pkl
    scratch_path: path containing hdf5/*.h5 (defaults to sim_path if not given)

    Memory layout:
      Phase 1: HDF5 scanned in CHUNK-row slices — peak ~1.5 GB.
      Phase 2: plume_all_af chunked — peak ~1.6 GB/chunk (was ~26 GB for full load).
      Phase 3: plume_all_mf chunked, mean_cb_mf accumulated on the fly — peak ~3 GB/chunk.
    All phases use CHUNK-row slices; suitable for a login node.
    """

    dx, dy, dz = 250, 250, 50   # m
    grid_vol   = dx * 1e-3 * dy * 1e-3 * dz * 1e-3  # km³
    cbl_gap    = 3
    qclm_crit  = 1.0e-6
    CHUNK      = 5000   # rows per HDF5 read; ~780 MB transient per iteration

    sim_path     = Path(sim_path)
    scratch_path = Path(scratch_path) if scratch_path is not None else sim_path

    with open(sim_path / 'uninterrupted_large_clouds.json') as f:
        ul_clouds = json.load(f)
    nc_json = len(ul_clouds)

    # Consistency check: NC must agree across all three HDF5 files and the JSON
    hdf5_files = {
        'cloud_all_af.h5': 'af',
        'plume_all_af.h5': 'af',
        'plume_all_mf.h5': 'mf',
    }
    nc_shapes = {}
    for fname, ds_key in hdf5_files.items():
        with h5py.File(scratch_path / 'hdf5' / fname) as f:
            nc_shapes[fname] = f[ds_key].shape[0]

    if len(set(nc_shapes.values())) > 1:
        raise ValueError(
            "NC mismatch across HDF5 files:\n" +
            "\n".join(f"  {k}: NC={v}" for k, v in nc_shapes.items())
        )
    nc_hdf5 = next(iter(nc_shapes.values()))
    if nc_json != nc_hdf5:
        raise ValueError(
            f"NC mismatch: uninterrupted_large_clouds.json has {nc_json} clouds "
            f"but HDF5 files have {nc_hdf5} rows"
        )

    # Phase 1: chunked scan — never holds the full (NC, NT, NZ) array in memory
    with h5py.File(scratch_path / 'hdf5/cloud_all_af.h5') as f:
        ds = f['af']
        nc, nt, nz = ds.shape

        cloud_volumes  = np.zeros((nc, nt), dtype=np.float32)
        area_z         = np.zeros((nc, nz), dtype=np.float32)
        max_cloud_area = np.zeros(nc,       dtype=np.float32)
        cbl_c          = np.zeros((nc, nt), dtype=np.int16)

        for i0 in range(0, nc, CHUNK):
            i1    = min(i0 + CHUNK, nc)
            chunk = ds[i0:i1]                              # (cs, NT, NZ) float32
            cloud_volumes[i0:i1]  = chunk.sum(axis=2)
            area_z[i0:i1]         = chunk.sum(axis=1)
            max_cloud_area[i0:i1] = chunk.max(axis=(1, 2))
            cbl_c[i0:i1]          = np.argmax(chunk > 0, axis=2).astype(np.int16)
            print(f"  Phase 1: {i1}/{nc} rows processed", end='\r', flush=True)
    print()

    # Derived quantities from the small accumulated arrays
    cloud_times         = (cloud_volumes > 0).sum(axis=1)
    cloud_ini_time      = np.argmax(cloud_volumes > 0, axis=1)
    total_cloud_volumes = cloud_volumes.sum(axis=1) * grid_vol

    has_area = area_z > 0
    mcbl_c   = np.argmax(has_area, axis=1)
    mctl_c   = (nz - 1) - np.argmax(has_area[:, ::-1], axis=1)
    del area_z, has_area

    with open(sim_path / 'pkl/qclm.pkl', 'rb') as f:
        qclm = np.asarray(pickle.load(f))
    mcbl = np.argmax(qclm > qclm_crit, axis=1)   # (NT,)

    attached_c   = (cloud_volumes > 0) & (cbl_c.astype(np.int32) - cbl_gap < mcbl[np.newaxis, :])
    del cbl_c
    attached_ind   = np.where(attached_c.any(axis=1))[0]
    attached_c_att = attached_c[attached_ind, :]
    del attached_c, cloud_volumes
    print(f"  Attached clouds: {len(attached_ind)}", flush=True)

    # pre-compute level indices (needed in Phase 3)
    t_idx = np.arange(nt)
    l_idx = np.clip(mcbl - 1, 0, nz - 3)    # (NT,)

    # Phase 2: plume area — chunked over attached rows (~1.6 GB/chunk)
    max_plume_area = np.zeros(len(attached_ind), dtype=np.float32)
    with h5py.File(scratch_path / 'hdf5/plume_all_af.h5') as f:
        for j0 in range(0, len(attached_ind), CHUNK):
            j1 = min(j0 + CHUNK, len(attached_ind))
            chk = f['af'][attached_ind[j0:j1]]        # (cs, NT, NZ)
            max_plume_area[j0:j1] = chk.max(axis=(1, 2))
            print(f"  Phase 2: {j1}/{len(attached_ind)} rows", end='\r', flush=True)
    print()

    # Phase 3: plume mass flux — chunked, accumulate mean_cb_mf on the fly (~1.5 GB/chunk)
    mean_cb_mf = np.zeros(len(attached_ind), dtype=np.float32)
    with h5py.File(scratch_path / 'hdf5/plume_all_mf.h5') as f:
        for j0 in range(0, len(attached_ind), CHUNK):
            j1 = min(j0 + CHUNK, len(attached_ind))
            chk = f['mf'][attached_ind[j0:j1]] * np.float32(dx * dy)
            mf_t_chk = (
                chk[:, t_idx, l_idx    ] +
                chk[:, t_idx, l_idx + 1] +
                chk[:, t_idx, l_idx + 2]
            ) * np.float32(1/3)
            att_chk     = attached_c_att[j0:j1]
            att_sum_chk = att_chk.sum(axis=1).astype(np.float32)
            mean_cb_mf[j0:j1] = np.where(
                att_sum_chk > 0,
                (att_chk * mf_t_chk).sum(axis=1) / np.maximum(att_sum_chk, np.float32(1)),
                np.float32(0),
            )
            print(f"  Phase 3: {j1}/{len(attached_ind)} rows", end='\r', flush=True)
    print()
    del attached_c_att

    mean_cb_mf_c = np.where(mean_cb_mf > np.float32(1.0), mean_cb_mf, np.float32(1.1))

    return (
        mean_cb_mf_c, mean_cb_mf,
        total_cloud_volumes[attached_ind], cloud_times[attached_ind],
        mcbl_c[attached_ind], mctl_c[attached_ind],
        cloud_ini_time[attached_ind], max_cloud_area[attached_ind],
        max_plume_area, attached_ind,
    )


if __name__ == '__main__':
    runs = {
        'ctl':  ('goamazon_2pulse.largedom.r20251008.rerun',
                 'goamazon_2pulse.largedom.r20251008.rerun'),
        'ehe1': ('goamazon_2pulse.largedom.ehe1.r20251030.rerun',
                 'goamazon_2pulse.largedom.ehe1.r20251030.rerun'),
    }

    for name, (sim, scratch) in runs.items():
        print(f"\n{'='*50}", flush=True)
        print(f"Processing {name}: {sim}", flush=True)
        print(f"  HDF5 source: {scratch}", flush=True)
        stats = calculate_csd(f'{sim}/', f'{scratch}/')
        out_path = f'{sim}/pkl/csd_stats.sparse.claude.v2.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump(stats, f)
        print(f"  Saved -> {out_path}", flush=True)

    print("\nDone.", flush=True)
