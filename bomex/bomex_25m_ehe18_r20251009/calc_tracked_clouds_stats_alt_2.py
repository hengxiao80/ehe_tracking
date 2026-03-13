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

    dvs_header = "/dvs_ro/cfs/cdirs/m1657/xiao169/ehe_tracking/bomex/bomex_25m_ehe18_r20251009"
    scratch_header = "/pscratch/sd/x/xiao169/ehe_tracking/bomex/bomex_25m_ehe18_r20251009"

    with open(f"{dvs_header}/uninterrupted_large_clouds.json", "r") as f:
        ul_clouds = json.load(f)
    ul_clouds_list = list(ul_clouds.keys())
    nc = len(ul_clouds_list)
    id_list = np.asarray(ul_clouds_list, dtype=np.int32)

    filelist = sorted(glob.glob(f"{dvs_header}/OUT_3D/*.nc"))
    shift = 60

    area_names = ["core", "cloud", "plume"]
    na = len(area_names)
    sample_name = "all"

    nz = 120

    print(f"nt = {nt}, na = {na}, nc = {nc}, nz = {nz}")

    print(f"{ti}: {filelist[ti+shift]}")

    qc = np.zeros((na, nc, nz), dtype=float)

    clouds_in = h5py.File(f"{dvs_header}/hdf5/clouds_{ti:08d}.h5", "r")
    clouds = {k: v for (k, v) in clouds_in.items() if k in ul_clouds_list}
    clouds_list = list(clouds.keys())

    print(f"Processing {len(clouds_list)} clouds ...")

    if len(clouds_list) > 0:
        sam = xr.open_dataset(filelist[ti+shift])
        qn = sam.QN
        _, _, ny, nx = qn.shape

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
                    field[c_ijk[:, 0], c_ijk[:, 1], c_ijk[:, 2]] = (
                        mask[0, c_ijk[:, 0], c_ijk[:, 1], c_ijk[:, 2]]
                        * qn.data[0, c_ijk[:, 0], c_ijk[:, 1], c_ijk[:, 2]]
                    )
                    qc[i_area, ic, :] = field.sum(axis=(1, 2))

        del field, mask, qn
        sam.close()
    clouds_in.close()

    for i_area, area_name in enumerate(area_names):
        with open(
            f"{scratch_header}/pkl/{area_name}_{sample_name}_qc.{ti:08d}.pkl", "wb"
        ) as f:
            pickle.dump(qc[i_area, :, :], f)

if __name__ == "__main__":

    nt = 241

    with Pool(processes=128) as pool:
        pool.map(process_t, range(nt))
