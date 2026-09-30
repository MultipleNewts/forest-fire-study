# %%
import os
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import ff_rules as fr

# %%
# setup vars##########
dim = 200
######################

# Make a folder to store frame images (you can look inside it to debug your plots)
os.makedirs("frames", exist_ok=True)

# Example: starting state of your forest board
# 0 = empty, 1 = tree, 2 = fire
board = np.random.choice([0, 1, 2], size=(dim, dim), p=[0.35, 0.649, 0.001])
# Color map: black=empty, green=tree, red=fire
cmap = plt.matplotlib.colors.ListedColormap(["black", "green", "red"])
# cmap = plt.matplotlib.colors.ListedColormap(["black", "gray", "white"])

frames = []  # list to store all frame images

# Loop over time steps (50 here, but change as you like)
for i in range(50):
    # prevents overwriting reference board
    new_board = np.zeros((dim, dim))
    # go through rows and columns
    for y, row in enumerate(board):
        for x, column in enumerate(row):
            # find new state from updating reference board
            new_board[y, x] = fr.update_cell(board, x, y, dim=dim)
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
