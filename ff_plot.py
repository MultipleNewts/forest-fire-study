# %%
import matplotlib.pyplot as plt
import numpy as np


# %%
def plot_tile_states(x, empty, trees, fires):
    """
        Plots progression of tile states throughout simulation

        Parameters
        ----------
        x : `array`
            the array indicating time steps, one data point is one step/tick
        empty : `array`
            the 1d-array containing the number of "empty" (0) tiles at during each step of the sim
        trees : `array`
            the 1d-array containing the number of "tree" (1) tiles at during each step of the sim
        fires : `array`
            the 1d-array containing the number of "fire" (2) tiles at during each step of the sim
    """
    plt.plot(x, empty, label="Empty Tiles", c="k")
    plt.plot(x, trees, label="Trees", c="green")
    plt.plot(x, fires, label="Fires", c="red")
    plt.xlabel("Time (ticks)")
    plt.ylabel("Number of tiles")
    plt.title("Progression of tile states")
    plt.legend(loc="upper right")
    plt.show()


# %%
def plot_minmax(x, means, maxes, label, step=20):
    """
        Plots progression of cluster average and maximum values (for either cluster size or radius)
        as the sim progress. Should only take spaced out datapoints to prevent correlation.

        Parameters
        ----------
        x : `array`
            the array indicating the time steps of the selected data
        means : `array`
            a 1d-array containing the mean values of the variable at each relevant time step
        maxes : `array`
            a 1d-array containing the max values of the variable at each relevant time step
        label : `str`
            the name of the relevant variable, for figure labelling
        step : `int`
            the spacing between each datapoint, for figure labelling
    """
    # AI genned
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # 1) Dotted vertical connectors between each mean and its max
    ax.vlines(x, means, maxes, colors="gray", linestyles="dotted", linewidth=1.5, zorder=1)

    # 2) Solid lines connecting the means together and the maxes together
    ax.plot(x, means, color="tab:blue", linestyle="-", linewidth=1.8, zorder=2)
    ax.plot(x, maxes, color="tab:red", linestyle="-", linewidth=1.8, zorder=2)

    # 3) Markers (different colours AND shapes), drawn on top
    ax.scatter(x, means, color="tab:blue", marker="o", s=60, zorder=3, label="Mean")
    ax.scatter(x, maxes, color="tab:red", marker="^", s=70, zorder=3, label="Max")

    # Labels, title, legend
    ax.set_xlabel("Time Step")
    ax.set_ylabel(f"{label} Value")
    ax.set_title(f"Mean and Max {label} Values Every {step} Ticks")
    ax.set_xticks(x)
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.show()


# %%
def plot_bin_hist(sizes, max=100):
    """
        Plots the number of occurences of each sized cluster

        Parameters
        ----------
        sizes : `array`
            an array of sizes, can be flat or inhomogenous, just needs all counted sizes
    """
    flat = np.concatenate(sizes)
    num_b = len(np.unique(flat))
    plt.hist(flat, num_b)
    plt.xlim([0, max])
    plt.xlabel("Cluster Size")
    plt.ylabel("No. of Occurences")
    plt.title("Occurence of Cluster Sizes")
    plt.show()
