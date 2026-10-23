# %%
import os
import numpy as np
import ff_run as frun
import ff_plot as fplot
import ff_filehandler as fh
import hoshen_kopelman as hk
import ff_fit as fit
import FF_MLE as mle
# import ff_rules as fr

# %%

# setup variables
dim = 150
dur = 2000
fpratio = 0.005
p = 0.05
# ###

# Make a folder to store frame images for graphics
os.makedirs("frames", exist_ok=True)

# 0 = empty, 1 = tree, 2 = fire
board = np.random.choice([0, 1, 2], size=(dim, dim), p=[0.5, 0.49, 0.01])

# %%
# choose how to get data
# frun.stand_run: runs sim with high_speed and no graphics
# frun.graphical_run: runs sim and generates animation - slower
# fh.use_stored("name"): uses saved data rather than running a new sim

data, bdata = frun.stand_run(board, dim, dur)

# %%
# if True, store this runs data under the specified name
store = False
if store is True:
    fh.write_data(bdata, dim, dur, "LargeRun")

# %%
# structures tile data for study
x_data = np.arange(len(data))
data = np.array(data)
empty, trees, fires = data.T

# plots tile data
fplot.plot_tile_states(x_data, empty, trees, fires)

# %%
data_step = 100
samples = bdata[::data_step]
sizs = []
rads = []
n_clusters = []
count_sz = []
incr = 0
incr_check = len(samples)//10
for sample in samples:
    incr += 1
    if incr % incr_check == 0:
        print(incr)
    tlabel, tlabels = hk.HK_cluster(sample, dim)
    tlabel = hk.match_labels(tlabel, tlabels, dim)

    clusters = np.unique(tlabel)[1:]
    n_clusters.append(len(clusters))
    centres, clusters = hk.locate_centres(clusters, tlabel, dim)

    sizes = []
    for cluster in clusters:
        sizes.append(len(cluster))

    radii = []
    for i, centre in enumerate(centres):
        radii.append(hk.compute_radius(clusters[i], centre))

    count_sz.append(sizes)
    mmrad = [np.max(radii), np.mean(radii)]
    mmsiz = [np.max(sizes), np.mean(sizes)]
    sizs.append(mmsiz)
    rads.append(mmrad)


# %%
check_val = np.mean(np.log(np.concatenate(count_sz)))
N = len(np.concatenate(count_sz))
alpha = mle.find_param(check_val, strt=2)
error = mle.get_error(alpha, N)
print(f"Paramter is {alpha:.4f} +- {error:.4f}")

# %%
fplot.plot_bin_hist(count_sz, alpha, error)

# %%
stemp = np.array(sizs).T
rtemp = np.array(rads).T
smax = stemp[0]
savg = stemp[1]
rmax = rtemp[0]
ravg = rtemp[1]
x = np.arange(len(smax), dtype=float)*data_step

# %%
epsilon = 1e-12
params = fit.fit_params(savg, x+epsilon, ntests=50, kran=[-4, 4], pran=[-4, 4], oran=[-10, 10])
fit_model = fit.model(x+epsilon, *params)
fit.plot_dat_mod(x, savg, params, y_label="Cluster Size")
print(params)
