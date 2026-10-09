# %%
import os
import numpy as np
import ff_run as frun
import ff_plot as fplot
# import matplotlib.pyplot as plt
# import ff_rules as fr

# %%
# setup vars##########
dim = 150
fpratio = 0.001
p = 0.001
######################

# Make a folder to store frame images (you can look inside it to debug your plots)
os.makedirs("frames", exist_ok=True)

# 0 = empty, 1 = tree, 2 = fire
board = np.random.choice([0, 1, 2], size=(dim, dim), p=[1, 0, 0])

# %%
# data = frun.graphical_run(board, dim, 100)
data = frun.stand_run(board, dim, 100)
# %%
x_data = np.arange(len(data))
data = np.array(data)
empty, trees, fires = data.T

fplot.plot_tile_states(x_data, empty, trees, fires)

# %%
