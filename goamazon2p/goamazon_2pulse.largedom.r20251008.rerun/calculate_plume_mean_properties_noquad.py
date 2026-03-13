import numpy as np
import xarray as xr
import pickle
import dask

from dask.distributed import Client

if __name__ == '__main__':
    dask.config.set(scheduler='threads', **{'array.slicing.split_large_chunks': True})
    # somehow without explicitly specifying 1 thread per worker, I get a lot of non-fatal errors out of xarray or netcdf4.
    client = Client(n_workers=128, threads_per_worker=1)

    with open("pkl/wm.pkl", "rb") as f:
        wm = pickle.load(f)
    with open("pkl/qtm.pkl", "rb") as f:
        qtm = pickle.load(f)

    # GoAmazon large domain file pattern
    times = np.arange(7200, 14401, 30)
    files = [f"OUT_3D/GOAMAZON_2pulse.large.r20251008_256_zapped_{t:010d}.nc" for t in times]
    sam = xr.open_mfdataset(files, parallel=True, data_vars='minimal', coords='minimal', compat='override')
    qt = sam.QV + sam.QN
    wi = sam.W
    w = (wi.shift(z=-1, fill_value=0.0) + wi) * 0.5
    qcl = sam.QN
    qta = qt - qtm
    wa = w - wm

    qta, wa, qcl = client.persist([qta, wa, qcl])

    ots = np.arange(0, 241)
    files = [f"tracked_volumes/tracked_volumes_{t:08d}.nc" for t in ots]
    tv = xr.open_mfdataset(files, parallel=True, data_vars='minimal', coords='minimal', compat='override')
    tracked_p = client.persist(tv.tracked_plume)

    ots = np.arange(0, 241)
    files = [f"tracked_volumes/tracked_volumes_large_{t:08d}.nc" for t in ots]
    tvl = xr.open_mfdataset(files, parallel=True, data_vars='minimal', coords='minimal', compat='override')
    tl_p = client.persist(tvl.tracked_plume)

    ots = np.arange(0, 241)
    files = [f"tracked_volumes/tracked_volumes_uninterrupted_large_{t:08d}.nc" for t in ots]
    tvul = xr.open_mfdataset(files, parallel=True, data_vars='minimal', coords='minimal', compat='override')
    tul_p = client.persist(tvul.tracked_plume)

    ots = np.arange(0, 241)
    files = [
        f"tracked_volumes/tracked_volumes_uninterrupted_large_attached_{t:08d}.nc"
        for t in ots
    ]
    tvula = xr.open_mfdataset(files, parallel=True, data_vars='minimal', coords='minimal', compat='override')
    tula_p = client.persist(tvula.tracked_plume)

    # Compute domain-wide quadrant statistics
    qfl = (wa * qta).mean(axis=(0, 2, 3))
    with open(f"pkl/qflux_dom.pkl", "wb") as f:
        pickle.dump(qfl.compute(), f)
    qqcm = qcl.mean(axis=(0, 2, 3)).compute()
    with open(f"pkl/qcm_dom.pkl", "wb") as f:
        pickle.dump(qqcm, f)

    # Compute statistics for tracked cloud populations
    p = [tracked_p, tl_p, tul_p, tula_p]
    pn = ["t", "tl", "tul", "tula"]

    for pp, ppn in zip(p, pn):
        qfl = (pp * wa * qta).mean(axis=(0, 2, 3))
        with open(f"pkl/qflux_{ppn}.pkl", "wb") as f:
            pickle.dump(qfl.compute(), f)
        qqcm = ((pp * qcl).mean(axis=(0, 2, 3))).compute()
        with open(f"pkl/qcm_{ppn}.pkl", "wb") as f:
            pickle.dump(qqcm, f)

    print("Computation complete!")
    client.close()
