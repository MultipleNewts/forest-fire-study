# %%
import os
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import ff_rules as fr
import ff_fitcurve as fc

# %%
# setup vars##########
dim = 150
fpratio = 0.001
p = 0.001
######################

# Make a folder to store frame images (you can look inside it to debug your plots)
os.makedirs("frames", exist_ok=True)

# Example: starting state of your forest board
# 0 = empty, 1 = tree, 2 = fire
board = np.random.choice([0, 1, 2], size=(dim, dim), p=[1, 0, 0])
# Color map: black=empty, green=tree, red=fire
cmap = plt.matplotlib.colors.ListedColormap(["black", "green", "red"])
# cmap = plt.matplotlib.colors.ListedColormap(["black", "gray", "white"])

frames = []  # list to store all frame images
data = []

# Loop over time steps (50 here, but change as you like)
for i in range(100):
    temp = board.ravel()
    data.append(np.bincount(temp, minlength=3))

    # prevents overwriting reference board
    new_board = np.zeros((dim, dim), dtype=int)
    # go through rows and columns
    for y, row in enumerate(board):
        for x, column in enumerate(row):
            # find new state from updating reference board
            new_board[y, x] = fr.update_cell(board, x, y, dim=dim, p=0.1, fpratio=0.1)
    # new reference board is updated board
    board = new_board

    # Plot the board
    plt.imshow(board, cmap=cmap, vmin=0, vmax=2)
    plt.title(f"Time: {i}")

    # Save the current frame as a PNG
    frame_path = os.path.join("frames", f"frame{i}.png")
    plt.savefig(frame_path)

    # Load the PNG back into memory for the GIF
    frames.append(iio.imread(frame_path))

    # Close the figure so we don’t pile up too many open plots
    plt.close()

# Save all frames into a looping GIF. Increase frames per second (fps) if needed
iio.imwrite("movie.gif", frames, loop=0, fps=10)
print("Saved movie.gif (check your folder!)")
# %%
x_data = np.arange(len(data))
data = np.array(data)
empty, trees, fires = data.T

plt.plot(x_data, empty, label="Empty Tiles", c="k")
plt.plot(x_data, trees, label="Trees", c="green")
plt.plot(x_data, fires, label="Fires", c="red")
plt.xlabel("Time (ticks)")
plt.ylabel("Number of tiles")
plt.title("Progression of tile states")
plt.legend(loc="upper right")
plt.show()

# %%
tree_fit, vars = fc.power_fit(x_data[10:], trees[10:])
plt.scatter(x_data[10:], trees[10:], label="Trees", c="lightgreen", linewidths=1, marker="x")
plt.plot(x_data[10:], tree_fit, label="Power Law Best Fit", c="green", linestyle="--")
plt.xlabel("Time (ticks)")
plt.ylabel("Number of tiles")
plt.title("Progression of tile states")
plt.legend(loc="upper right")
plt.show()
print(f"The graph has the form a*(x**k)+b where:\na={vars[0]}\nk={vars[1]}\nb={vars[2]}")

# %%
fire_fit, vars = fc.power_fit(x_data[10:], fires[10:])
plt.scatter(x_data[10:], fires[10:], label="Fires", c="darkred", linewidths=1, marker="x")
plt.plot(x_data[10:], fire_fit, label="Power Law Best Fit", c="red", linestyle="--")
plt.xlabel("Time (ticks)")
plt.ylabel("Number of tiles")
plt.title("Progression of tile states")
plt.legend(loc="upper right")
plt.show()
print(f"The graph has the form a*(x**k)+b where:\na={vars[0]}\nk={vars[1]}\nb={vars[2]}")

# %%
