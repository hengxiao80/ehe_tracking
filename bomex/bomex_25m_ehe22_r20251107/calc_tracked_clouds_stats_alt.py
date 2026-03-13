import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import h5py
import json
import pickle
import glob
from multiprocessing import Pool


def index_to_zyx(index, nx, ny):
    z = np.floor_divide(index, (ny * nx))
    index = np.mod(index, (ny * nx))
    y = np.floor_divide(index, (nx))
    x = np.mod(index, nx)
    return (z, y, x)


def process_t(ti):

    dvs_header = "/dvs_ro/cfs/cdirs/m1657/xiao169/ehe_tracking/bomex/bomex_25m_ehe22_r20251107"
    scratch_header = "/pscratch/sd/x/xiao169/ehe_tracking/bomex/bomex_25m_ehe22_r20251107"

    with open(f"{dvs_header}/uninterrupted_large_clouds.json", "r") as f:
        ul_clouds = json.load(f)
    ul_clouds_list = list(ul_clouds.keys())
    nc = len(ul_clouds_list)
    id_list = np.asarray(ul_clouds_list, dtype=np.int32)

    with open(f"{dvs_header}/pkl/wm.pkl", "rb") as f:
        wm = pickle.load(f)
    with open(f"{dvs_header}/pkl/ws.pkl", "rb") as f:
        ws = pickle.load(f)
    with open(f"{dvs_header}/pkl/qtm.pkl", "rb") as f:
        qtm = pickle.load(f)
    with open(f"{dvs_header}/pkl/qts.pkl", "rb") as f:
        qts = pickle.load(f)
    with open(f"{dvs_header}/pkl/tvm.pkl", "rb") as f:
        tvm = pickle.load(f)

    filelist = sorted(glob.glob(f"{dvs_header}/OUT_3D/*.nc"))
    shift = 60

    area_names = ["core", "cloud", "plume"]
    na = len(area_names)
    sample_name = "all"

    nz = 120

    print(f"nt = {nt}, na = {na}, nc = {nc}, nz = {nz}")

    print(f"{ti}: {filelist[ti+shift]}")

    af = np.zeros((na, nc, nz), dtype=float)
    wq = np.zeros((na, nc, nz), dtype=float)
    mf = np.zeros((na, nc, nz), dtype=float)

    clouds_in = h5py.File(f"{dvs_header}/hdf5/clouds_{ti:08d}.h5", "r")
    clouds = {k: v for (k, v) in clouds_in.items() if k in ul_clouds_list}
    clouds_list = list(clouds.keys())

    print(f"Processing {len(clouds_list)} clouds ...")

    if len(clouds_list) > 0:
        sam = xr.open_dataset(filelist[ti+shift])
        wi = sam.W
        w = (wi.shift(z=-1, fill_value=0.0) + wi) * 0.5
        wa = w - wm
        wn = wa / ws
        qt = sam.QV + sam.QN
        qta = qt - qtm
        qtn = qta / qts
        # tr = sam.TR01
        # trn = (tr - trm)/trs
        # tv = sam.TABS * (1.0 + 0.608 * sam.QV * 0.001 - sam.QN * 0.001)
        # tva = tv - tvm
        infile2 = xr.open_dataset(f"{dvs_header}/density/density_{ti:08d}.nc")
        rho = infile2.density.data
        _, _, ny, nx = qtn.shape

        # sample_names = ["all", "mu", "md", "mupb", "munb", "mdpb", "mdnb"]
        # sample_masks = [
        #     np.ones((1, nz, ny, nx)),
        #     ((wn > 0.0) & (qtn > 0.0) & (wn * qtn > 1.0)).data,
        #     ((wn < 0.0) & (qtn > 0.0) & (wn * qtn < -1.0)).data,
        #     ((wn > 0.0) & (qtn > 0.0) & (wn * qtn > 1.0) & (tva > 0.0)).data,
        #     ((wn > 0.0) & (qtn > 0.0) & (wn * qtn > 1.0) & (tva <= 0.0)).data,
        #     ((wn < 0.0) & (qtn > 0.0) & (wn * qtn < -1.0) & (tva > 0.0)).data,
        #     ((wn < 0.0) & (qtn > 0.0) & (wn * qtn < -1.0) & (tva <= 0.0)).data,
        # ]
        mask = np.ones((1, nz, ny, nx)) 

        field = np.zeros((nz, ny, nx), dtype=float)
        for id in clouds_list:
            ic = np.where(id_list == int(id))[0][0]
            # print(f"ti = {ti}: processing cloud {id} ...")
            for i_area, area in enumerate(["core", "condensed", "plume"]):
                c = np.asarray(clouds[id][area])
                if len(c) > 0:
                    c_ijk = np.asarray([index_to_zyx(ijk, ny, nx) for ijk in c])
                    field[:] = 0.0
                    field[c_ijk[:, 0], c_ijk[:, 1], c_ijk[:, 2]] = mask[
                        0, c_ijk[:, 0], c_ijk[:, 1], c_ijk[:, 2]
                    ]
                    af[i_area, ic, :] = field.sum(axis=(1, 2))

                    field[:] = 0.0
                    field[c_ijk[:, 0], c_ijk[:, 1], c_ijk[:, 2]] = (
                        mask[0, c_ijk[:, 0], c_ijk[:, 1], c_ijk[:, 2]]
                        * wn.data[0, c_ijk[:, 0], c_ijk[:, 1], c_ijk[:, 2]]
                        * qtn.data[0, c_ijk[:, 0], c_ijk[:, 1], c_ijk[:, 2]]
                    )
                    wq[i_area, ic, :] = field.sum(axis=(1, 2))

                    field[:] = 0.0
                    field[c_ijk[:, 0], c_ijk[:, 1], c_ijk[:, 2]] = (
                        mask[0, c_ijk[:, 0], c_ijk[:, 1], c_ijk[:, 2]]
                        * wa.data[0, c_ijk[:, 0], c_ijk[:, 1], c_ijk[:, 2]]
                        * rho[0, c_ijk[:, 0], c_ijk[:, 1], c_ijk[:, 2]]
                    )
                    mf[i_area, ic, :] = field.sum(axis=(1, 2))

        del field, mask, wi, w, wn, wa, qt, qta, qtn, rho
        sam.close()
        infile2.close()
    clouds_in.close()

    for i_area, area_name in enumerate(area_names):
        with open(
            f"{scratch_header}/pkl/{area_name}_{sample_name}_af.{ti:08d}.pkl", "wb"
        ) as f:
            pickle.dump(af[i_area, :, :], f)
        with open(
            f"{scratch_header}/pkl/{area_name}_{sample_name}_wq.{ti:08d}.pkl", "wb"
        ) as f:
            pickle.dump(wq[i_area, :, :], f)
        with open(
            f"{scratch_header}/pkl/{area_name}_{sample_name}_mf.{ti:08d}.pkl", "wb"
        ) as f:
            pickle.dump(mf[i_area, :, :], f)


if __name__ == "__main__":

    nt = 241

    with Pool(processes=128) as pool:
        pool.map(process_t, range(nt))
