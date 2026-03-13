import numpy as np
import xarray as xr
from datetime import datetime, timedelta
# from scipy import stats
import h5py
import json

def index_to_zyx(index, nx, ny):
    z = np.floor_divide(index, (ny * nx))
    index = np.mod(index, (ny * nx))
    y = np.floor_divide(index, (nx))
    x = np.mod(index, nx)
    return (z, y, x)

def tracked(clouds, input):
    nz, ny, nx = input.condensed.shape[1:]
    # id = input.core.copy()
    core_id = xr.zeros_like(input.core, dtype=np.int32)
    cloud_id = xr.zeros_like(input.condensed, dtype=np.int32)
    plume_id = xr.zeros_like(input.plume, dtype=np.int32)
    core_id.values[:] = -9999
    cloud_id.values[:] = -9999
    plume_id.values[:] = -9999
    for id in clouds.keys():
        for id_out, area in zip([core_id, cloud_id, plume_id], ['core', 'condensed', 'plume']):
            c = np.asarray(clouds[id][area])
            if len(c) > 0:
                c_ijk = np.asarray([index_to_zyx(ijk, ny, nx) for ijk in c])
                id_out.values[0, c_ijk[:,0], c_ijk[:,1], c_ijk[:,2]] = id
    return core_id, cloud_id, plume_id

for ti in np.arange(0, 241):
    print(ti)
    clouds_in = h5py.File(f'hdf5/clouds_{ti:08d}.h5', 'r')
    input = xr.open_dataset(f'data/cloudtracker_input_{ti:08d}.nc')
    ds = xr.Dataset(coords= {'time': input.time, 'z': input.z, 'y':input.y, 'x':input.x}) 
    core_id, cloud_id, plume_id = tracked(clouds_in, input)
    ds['core_id'] = core_id 
    ds['cloud_id'] = cloud_id 
    ds['plume_id'] = plume_id 
    ds.to_netcdf(f'cloud_ids/cloud_ids_{ti:08d}.nc')
    del clouds_in, input, ds, core_id, cloud_id, plume_id



