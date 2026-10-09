# %%
import os
import numpy as np
import ff_run as frun
import ff_plot as fplot
import ff_filehandler as fh
import hoshen_kopelman as hk
# import matplotlib.pyplot as plt
# import ff_rules as fr

# %%

# setup variables
dim = 150
dur = 100
fpratio = 0.001
p = 0.001

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

# if True, store this runs data under the specified name
store = False
if store is True:
    fh.write_data(bdata, dim, dur, "test")

# %%
# structures tile data for study
x_data = np.arange(len(data))
data = np.array(data)
empty, trees, fires = data.T

# plots tile data
fplot.plot_tile_states(x_data, empty, trees, fires)

# %%
samples = bdata[::20]
samp_data = []
for sample in samples:
    tlabel, tlabels = hk.HK_cluster(sample, dim)
    tlabel = hk.match_labels(tlabel, tlabels, dim)

    clusters = np.unique(tlabel)[1:]
    centres, clusters = hk.locate_centres(clusters, tlabel, dim)

    sizes = []
    for cluster in clusters:
        sizes.append(len(cluster))

    radii = []
    for i, centre in enumerate(centres):
        radii.append(hk.compute_radius(clusters[i], centre))

    samp_data.append([sizes, radii])

# %%
