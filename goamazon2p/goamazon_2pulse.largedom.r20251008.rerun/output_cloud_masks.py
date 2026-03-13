import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import h5py
import json
import pickle

def index_to_zyx(index, nx, ny):
    z = np.floor_divide(index, (ny * nx))
    index = np.mod(index, (ny * nx))
    y = np.floor_divide(index, (nx))
    x = np.mod(index, nx)
    return (z, y, x)

def tracked(clouds_in, input, subset=[]):
    nz, ny, nx = input.condensed.shape[1:]
    tracked_core = input.core.copy()
    tracked_cloud = input.condensed.copy()
    tracked_plume = input.plume.copy()
    tracked_core.values[:] = False
    tracked_cloud.values[:] = False
    tracked_plume.values[:] = False
    if subset:
        clouds = {k:v for (k,v) in clouds_in.items() if k in subset} 
    else:
        clouds = clouds_in
    for id in clouds.keys():
        for area in ['core', 'condensed', 'plume']:
            c = np.asarray(clouds[id][area])
            if len(c) > 0:
                c_ijk = np.asarray([index_to_zyx(ijk, ny, nx) for ijk in c])
                tracked_core.values[0, c_ijk[:,0], c_ijk[:,1], c_ijk[:,2]] = True
                tracked_cloud.values[0, c_ijk[:,0], c_ijk[:,1], c_ijk[:,2]] = True
                tracked_plume.values[0, c_ijk[:,0], c_ijk[:,1], c_ijk[:,2]] = True
    return tracked_core, tracked_cloud, tracked_plume

with open('hdf5/events.json', 'r') as f:
    all_clouds = json.load(f)

with open('large_clouds.json', 'r') as f:
    large_clouds = json.load(f)
large_clouds_list = list(large_clouds.keys())

with open('uninterrupted_large_clouds.json', 'r') as f:
    uninterrupted_large_clouds = json.load(f)
uninterrupted_large_clouds_list = list(uninterrupted_large_clouds.keys())

with open('pkl/csd_stats.pkl', 'rb') as f:
    csd_stats = pickle.load(f)
    attached_ind = csd_stats[-1]
ula_clouds = [uninterrupted_large_clouds_list[int(i)] for i in attached_ind]

print(f"# of clouds: {len(all_clouds)}")
print(f"# of large clouds: {len(large_clouds)}")
print(f"# of uninterrupted large clouds: {len(uninterrupted_large_clouds)}")
print(f"# of uninterrupted large and attached clouds: {len(ula_clouds)}")

nt = 241

print(f"# of clouds: {len(all_clouds)}")
for ti in np.arange(0, nt):
    clouds_in = h5py.File(f'hdf5/clouds_{ti:08d}.h5', 'r')
    input = xr.open_dataset(f'data/cloudtracker_input_{ti:08d}.nc')
    ds = xr.Dataset(coords= {'time': input.time, 'z': input.z, 'y':input.y, 'x':input.x}) 
    tracked_core, tracked_cloud, tracked_plume = tracked(clouds_in, input)
    ds['tracked_core'] = tracked_core
    ds['tracked_cloud'] = tracked_cloud
    ds['tracked_plume'] = tracked_plume
    ds.to_netcdf(f'tracked_volumes/tracked_volumes_{ti:08d}.nc')
    del clouds_in, input, ds, tracked_core, tracked_cloud, tracked_plume

print(f"# of large clouds: {len(large_clouds)}")
for ti in np.arange(0, nt):
    clouds_in = h5py.File(f'hdf5/clouds_{ti:08d}.h5', 'r')
    input = xr.open_dataset(f'data/cloudtracker_input_{ti:08d}.nc')
    ds = xr.Dataset(coords= {'time': input.time, 'z': input.z, 'y':input.y, 'x':input.x}) 
    tracked_core, tracked_cloud, tracked_plume = tracked(clouds_in, input, subset=large_clouds_list)
    ds['tracked_core'] = tracked_core
    ds['tracked_cloud'] = tracked_cloud
    ds['tracked_plume'] = tracked_plume
    ds.to_netcdf(f'tracked_volumes/tracked_volumes_large_{ti:08d}.nc')
    del clouds_in, input, ds, tracked_core, tracked_cloud, tracked_plume

print(f"# of uninterrupted large clouds: {len(uninterrupted_large_clouds)}")
for ti in np.arange(0, nt):
    clouds_in = h5py.File(f'hdf5/clouds_{ti:08d}.h5', 'r')
    input = xr.open_dataset(f'data/cloudtracker_input_{ti:08d}.nc')
    ds = xr.Dataset(coords= {'time': input.time, 'z': input.z, 'y':input.y, 'x':input.x}) 
    tracked_core, tracked_cloud, tracked_plume = tracked(clouds_in, input, subset=uninterrupted_large_clouds_list)
    ds['tracked_core'] = tracked_core
    ds['tracked_cloud'] = tracked_cloud
    ds['tracked_plume'] = tracked_plume
    ds.to_netcdf(f'tracked_volumes/tracked_volumes_uninterrupted_large_{ti:08d}.nc')
    del clouds_in, input, ds, tracked_core, tracked_cloud, tracked_plume

print(f"# of uninterrupted large and attached clouds: {len(ula_clouds)}")
for ti in np.arange(0, nt):
    clouds_in = h5py.File(f'hdf5/clouds_{ti:08d}.h5', 'r')
    input = xr.open_dataset(f'data/cloudtracker_input_{ti:08d}.nc')
    ds = xr.Dataset(coords= {'time': input.time, 'z': input.z, 'y':input.y, 'x':input.x}) 
    tracked_core, tracked_cloud, tracked_plume = tracked(clouds_in, input, subset=ula_clouds)
    ds['tracked_core'] = tracked_core
    ds['tracked_cloud'] = tracked_cloud
    ds['tracked_plume'] = tracked_plume
    ds.to_netcdf(f'tracked_volumes/tracked_volumes_uninterrupted_large_attached_{ti:08d}.nc')
    del clouds_in, input, ds, tracked_core, tracked_cloud, tracked_plume

print("Completed generating all tracked volume files!")
