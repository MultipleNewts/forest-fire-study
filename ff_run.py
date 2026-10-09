# %%
import os
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import ff_rules as fr


# %%
def graphical_run(board, dim, dur):
    # Color map: black=empty, green=tree, red=fire
    cmap = plt.matplotlib.colors.ListedColormap(["black", "green", "red"])
    # cmap = plt.matplotlib.colors.ListedColormap(["black", "gray", "white"])

    data = []
    frames = []
    # Loop over time steps
    for i in range(dur):
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

    return data


def stand_run(board, dim, dur):
    tile_data = []
    board_data = []

    # Loop over time steps
    for i in range(dur):
        temp = board.ravel()
        board_data.append(board)
        tile_data.append(np.bincount(temp, minlength=3))

        # prevents overwriting reference board
        new_board = np.zeros((dim, dim), dtype=int)
        # go through rows and columns
        for y, row in enumerate(board):
            for x, column in enumerate(row):
                # find new state from updating reference board
                new_board[y, x] = fr.update_cell(board, x, y, dim=dim, p=0.1, fpratio=0.1)
        # new reference board is updated board
        board = new_board
    return tile_data, board_data
