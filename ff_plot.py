# %%
import matplotlib.pyplot as plt


# %%
def plot_tile_states(x, empty, trees, fires):
    plt.plot(x, empty, label="Empty Tiles", c="k")
    plt.plot(x, trees, label="Trees", c="green")
    plt.plot(x, fires, label="Fires", c="red")
    plt.xlabel("Time (ticks)")
    plt.ylabel("Number of tiles")
    plt.title("Progression of tile states")
    plt.legend(loc="upper right")
    plt.show()
