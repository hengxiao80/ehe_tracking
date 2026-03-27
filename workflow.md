# Overall workflow

## SAM LES runs

- The usual SAM code at `/global/homes/x/xiao169/sam4clubb` is used. The branch is `env_homogenization`.

- Currently focusing on two cases,
    - BOMEX (quasi-steady marine shallow cumulus, 6-hour run) 
        - dx=dy=dz=25 m, Nx=Ny=512, Nz=120
    - The GOAMAZON double-pulse case from Tian and Zhang (2025, GRL). This is a shallow-to-deep transition case from the GOAMAZON campaign. The date is 2015-08-26. The run starts at 6AM local time and goes for 12 hours. We are interested in the shallow convection period (before precipitation becomes important).
        - Several different setups have been done:
            - dx=dy=50 m, dz=50 m up to 5 km, Nx=Ny=256
            - dx=dy=250 m, dz=50 m up to 5 km, Nx=Ny=256
            - dx=dy=250 m, dz=50 m up to 5 km, Nx=Ny=1024
            - There are also several other test runs performed because we found strong sensitivity to resolution in terms of cloud and precip. statistics.

- Other than the control run, we performed so-called Environment Homogenization (EH) runs for each case, where we smooth out environmental moisture variability in the environment, i.e., outside of the convectively active volumes.
    - The convectively active volume is where the surface-emitted scalar mixing ratio is higher than the horizontal mean plus $n$ times the horizontal standard deviation at a given time and vertical level. $n$ is a constant set to be 0.5. The smaller $n$ is the larger the convectively active volume is. 
    - The surface emitted scalar is totally passive and decays with a given time scale (30 minutes). For shallow convection, this definition of the convectively active volume is supposed to capture not just the clouds but the volume that is still influenced by convective mixing associated with shallow clouds (which are driven from the surface).
    - A time scale for the "smoothing" can be set. For the runs we focused on, that time scale is just one model time step. In other words, it is almost instantaneous.
    - For BOMEX, we apply EH only for the last 2 hours of the run in all the EH experiments.
        - We apply EH,
            1. In the entire cloud layer (from the mean cloud base all the way to a fixed level above the cloud layer) in `ehe18`;
            2. In the lower cloud layer only (~200 m) in `ehe22`;
            3. In the upper cloud layer only in `ehe21`.
    - For GOAMAZON, we apply EH only for 2 hours during the shallow convection period. We apply EH for the whole cloud layer.

- 30-second 3-D snapshots are saved for the 2-hour period for cloud tracking

- Postprocessing
    - For GOAMAZON cases, a "zapped" version of the 3-D output is generated where only the lower layers where 'dz' is uniform (50 m) are kept. The upper layers are deleted. This is because the cloud tracker cannot handle non-uniform vertical grid.

- Analyses

The main working directory for setup and basic performance analyses of the runs is `/global/cfs/cdirs/m1657/xiao169/env_homo_exps`.

The runs are performed using the scratch space (`/pscratch/sd/x/xiao169/sam_runs`). After they were done, I staged them on HPSS at `/home/x/xiao169/bomex_ehe` and `/home/x/xiao169/goamazon_ehe` respectively.

## Cloud tracking analyses

The cloud_tracker code is on the common/software partition at `/global/common/software/m1657/xiao169/cloud_tracker`. All the data to/from the analyses is at `/global/cfs/cdirs/m1657/xiao169/cloud_tracker_data`.

## Analyzing the impact of Environmental Homogenization on the shallow cloud population

This is done at the current directory (`/global/cfs/cdirs/m1657/xiao169/ehe_tracking`). Here is a list of analyses I have done for each case so far,

### BOMEX

Scripts present in each `run`:

- `filter_tracked_clouds.ipynb`: filter out smaller clouds for subsequent analyses. Use `.json` files to store filtered cloud info

- `calc_density.ipynb`: calculate density fields for every snapshot

- `calc_mean_var.ipynb`: calculates the 2-hour mean and variance profiles. Results stored in `pkl`

- `calc_tracked_clouds_stats_alt_2.py`/`calc_tracked_clouds_stats_alt.py`: calculate "core", "condensed", "plume" statistcs for `qc`/`af`, `wq`, `mf`. `calc_tracked_clouds_stats_alt*.slurm` is the submission script for both. `combine_pkl_t.ipynb` is used to glue the per-time output together and put into `pkl`.


- `output_cloud_masks.py`: output the 3-D  masks of various tracked cloud populaions to `tracked_volumes`.

- `calculate_plume_mean_properties.py`: calculates area fraction, normalized qflux, normalized w', normalized q' and qc for MU, MD, DD, DU and EN for different various tracked populations. Stores output in `pkl`. Only done for control and ehe18. Uses data from `{run}/tracked_volumes/`. The output from this script is mainly used by `total_qc_diff.ehe18.ipynb`/`total_wq_diff.ehe18.ipynb` to confirm that the change in w'q' and qc in our tracked/attached cloud population is consistent with the overall whole domain change.

- `output_tracked_cloud_ids.py`: output for each cloud its mask (as id) in 3-D for each time step. Not used for this paper yet. Only present in control.

- `csd.ipynb`: old notebook for calculating CSD for each run. Only present in control and ehe18.

- `moist_halo_vs_plume.p1.ipynb`: quicklook examination of the differences between our plume and the moist halo used by, e.g., Gu et al. Only present in control.

Scripts for the whole set of runs:

- `calc_csd.ipynb`: calculate and store CSD for each run being analyzed.

- `compare_csd.ipynb`: compare the CSDs in different ways.

- `compare_bin_lifetime.ipynb`: compare bin-averaged cloud top height and cloud lifetime for the whole set.

- `compare_bin_profiles.ipynb`: compare bin-averaged cloud qc and w'q' profiles.

- `compare_bin_profiles.2.ipynb`: compare bin-averaged bin-summed massflux profiles.

- `total_qc_diff.ehe18.ipynb`/`total_wq_diff.ehe18.ipynb`: compare the qc/w'q' differences between control and ehe18 across different tracked populations for consistency check. In other words, check if the change in the fully tracked & attached population that we analyze can explain the overall difference between the two runs. Quick answer is YES. Uses output from `calculate_plume_mean_properties.py`.

### GOAMAZON2p

I am analyzing three pairs of experiments for this case:

- Small-domain high-res pair: `goamazon_2pulse.smalldom.r20251008.rerun` and `goamazon_2pulse.smalldom.ehe1.r20251030.rerun`

- Large-domain low-res pair: `goamazon_2pulse.largedom.r20251008.rerun` and `goamazon_2pulse.largedom.ehe1.r20251030.rerun`

- Very-large-domain low-res pair: `goamazon_2pulse.largedom_1024.r20260116` and `goamazon_2pulse.largedom_1024.ehe1.r20260115`

The analysis scripts/notebooks in each run directory are basically the same as those in the BOMEX runs except that,

- `calculate_plume_mean_properties_noquad.py` is used instead of `calculate_plume_mean_properties.py` for efficiency.

- `calc_tracked_clouds_stats_alt_sparse.py` is the updated version (thanks to Claude) that runs much faster of `calc_tracked_clouds_stats_alt.py`. `combine_pkl_t_sparse.py` now produces the cloud stats files (.h5) and stores them in `hdf5/` instead of `pkl`. [NOT yet used for small_dom runs]

Scripts for comparing pairs of simulations:

These are similar to what we have for BOMEX.

- `calc_csd.*.ipynb`/`calc_csd_sparse.*.ipynb`: 
- `ehe1_vs_ctl.csd.*.ipynb`:  
- `compare_bin_lifetime.*.ipynb`: 
- `compare_bin_profiles.*.ipynb`
- `total_qc_diff.ehe1.*.ipynb`/`total_wq_diff.ehe1.*.ipynb`

There are also earlier versions of `compare_bin_profiles.*`:

- `bin_mean_wq_diff.*`
- `bin_qc_diff.*`
- `bin_wq_diff.*`



