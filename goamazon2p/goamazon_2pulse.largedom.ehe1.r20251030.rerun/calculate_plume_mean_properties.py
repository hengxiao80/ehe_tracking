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
    with open("pkl/ws.pkl", "rb") as f:
        ws = pickle.load(f)
    with open("pkl/trm.pkl", "rb") as f:
        trm = pickle.load(f)
    with open("pkl/trs.pkl", "rb") as f:
        trs = pickle.load(f)
    with open("pkl/qtm.pkl", "rb") as f:
        qtm = pickle.load(f)
    with open("pkl/qts.pkl", "rb") as f:
        qts = pickle.load(f)
    with open("pkl/tvm.pkl", "rb") as f:
        tvm = pickle.load(f)
    with open("pkl/qclm.pkl", "rb") as f:
        qclm = pickle.load(f)

    # GoAmazon large domain file pattern
    times = np.arange(0, 24001, 100)
    files = [f"OUT_3D/GOAMAZON_2pulse_0_{t:010d}.nc" for t in times]
    sam = xr.open_mfdataset(files, parallel=True, data_vars='minimal', coords='minimal', compat='override')
    qt = sam.QV + sam.QN
    wi = sam.W
    w = (wi.shift(z=-1, fill_value=0.0) + wi) * 0.5
    qcl = sam.QN
    tr = sam.TR01
    tv = sam.TABS * (1.0 + 0.608 * sam.QV * 0.001 - sam.QN * 0.001)
    tva = tv - tvm
    qtn = (qt - qtm) / qts
    wn = (w - wm) / ws
    wa = w - wm
    trn = (tr - trm) / trs

    qtn, wn, qcl = client.persist([qtn, wn, qcl])

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

    # Define quadrants: MU (moist updraft), MD (moist downdraft), DD (dry downdraft), DU (dry updraft), EN (environment)
    quad = [
        (qtn > 0.0) & (wn > 0.0) & (qtn * wn > 1.0),
        (qtn > 0.0) & (wn < 0.0) & (qtn * wn < -1.0),
        (qtn < 0.0) & (wn < 0.0) & (qtn * wn > 1.0),
        (qtn < 0.0) & (wn > 0.0) & (qtn * wn < -1.0),
        (qtn * wn >= -1.0) & (qtn * wn <= 1.0),
    ]
    quadname = ["MU", "MD", "DD", "DU", "EN"]
    
    # Compute domain-wide quadrant statistics
    for n, qd in zip(quadname, quad):
        qfr = qd.mean(axis=(0, 2, 3))
        with open(f"pkl/{n}_a_frac.pkl", "wb") as f:
            pickle.dump(qfr.compute(), f)
        qfl = (qd * wn * qtn).mean(axis=(0, 2, 3))
        with open(f"pkl/{n}_a_qflux.pkl", "wb") as f:
            pickle.dump(qfl.compute(), f)
        qwnm = (qd * wn).mean(axis=(0, 2, 3))
        with open(f"pkl/{n}_a_wnm.pkl", "wb") as f:
            pickle.dump(qwnm.compute(), f)
        qqtnm = (qd * qtn).mean(axis=(0, 2, 3))
        with open(f"pkl/{n}_a_qtnm.pkl", "wb") as f:
            pickle.dump(qqtnm.compute(), f)
        qqcm = ((qd * qcl).mean(axis=(0, 2, 3))).compute()
        with open(f"pkl/{n}_a_qcm.pkl", "wb") as f:
            pickle.dump(qqcm, f)

    # Compute statistics for tracked cloud populations
    p = [tracked_p, tl_p, tul_p, tula_p]
    pn = ["t", "tl", "tul", "tula"]

    for pp, ppn in zip(p, pn):
        quad = [
            (qtn > 0.0) & (wn > 0.0) & (qtn * wn > 1.0) & pp,
            (qtn > 0.0) & (wn < 0.0) & (qtn * wn < -1.0) & pp,
            (qtn < 0.0) & (wn < 0.0) & (qtn * wn > 1.0) & pp,
            (qtn < 0.0) & (wn > 0.0) & (qtn * wn < -1.0) & pp,
            (qtn * wn >= -1.0) & (qtn * wn <= 1.0) & pp,
        ]
        quadname_pop = [f"MU_a{ppn}", f"MD_a{ppn}", f"DD_a{ppn}", f"DU_a{ppn}", f"EN_a{ppn}"]
        for n, qd in zip(quadname_pop, quad):
            qfr = qd.mean(axis=(0, 2, 3))
            with open(f"pkl/{n}_frac.pkl", "wb") as f:
                pickle.dump(qfr.compute(), f)
            qfl = (qd * wn * qtn).mean(axis=(0, 2, 3))
            with open(f"pkl/{n}_qflux.pkl", "wb") as f:
                pickle.dump(qfl.compute(), f)
            qwnm = (qd * wn).mean(axis=(0, 2, 3))
            with open(f"pkl/{n}_wnm.pkl", "wb") as f:
                pickle.dump(qwnm.compute(), f)
            qqtnm = (qd * qtn).mean(axis=(0, 2, 3))
            with open(f"pkl/{n}_qtnm.pkl", "wb") as f:
                pickle.dump(qqtnm.compute(), f)
            qqcm = ((qd * qcl).mean(axis=(0, 2, 3))).compute()
            with open(f"pkl/{n}_qcm.pkl", "wb") as f:
                pickle.dump(qqcm, f)

    print("Computation complete!")
    client.close()
