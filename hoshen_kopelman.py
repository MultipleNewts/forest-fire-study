# %%
import numpy as np
import matplotlib.pyplot as plt


# ### TESTING VARS ### #
dim = 20
test_grid = np.random.choice([0, 1, 2], size=(dim, dim), p=[0.5, 0.49, 0.01])
# #################### #


def HK_cluster(ref_grid, dim):
    """
        Uses the Hoshen-Kopelman algorithm to idnetify clusters

        Parameters
        ----------
        grid : `array`
            the 2d array containing the board states
        dim : `int`
            the dimension of the board

        Returns
        -------
        grid : `array`
            a 2d-array containing labelled clusters (Note: clusters are matched through roots,
            so they may not directly show the correct number in this array.
            Run `match_labels(...)` to fix).
        labels : `array`
            a 1d-array containing the root of each index. Reduces time waste from doubling-back.
    """
    largest_label = 0
    grid = np.zeros((dim, dim), dtype=int)
    labels = np.arange(dim*dim)

    # perform raster scan
    for row in range(dim):
        for col in range(dim):
            if ref_grid[row][col] == 1:
                try:
                    left = grid[row-1][col]
                except TypeError:
                    left = 0
                try:
                    above = grid[row][col-1]
                except TypeError:
                    above = 0
                if (left == 0) and (above == 0):
                    largest_label += 1
                    grid[row][col] = largest_label
                elif (left != 0) and (above == 0):
                    grid[row][col] = find(left, labels)
                elif (left == 0) and (above != 0):
                    grid[row][col] = find(above, labels)
                else:
                    labels = union(left, above, labels)
                    grid[row][col] = find(left, labels)

    return grid, labels[:largest_label+1]


def find(value, labels):
    """
        Finds the root of the current cell

        Parameters
        ----------
        value : `int`
            the cluster index of the cell
        labels : `array`
            a 1d-array containing the root of each index

        Returns
        -------
        root : `int`
            the root index of the current cell. Ensures clusters are matched properly
    """
    if labels[value] != value:
        return find(labels[value], labels)
    return labels[value]


def union(cell_l, cell_a, labels):
    """
        Updates the root of a cluster to bind two clusters

        Parameters
        ----------
        cell_l : `int`
            the cluster index of the left cell
        cell_a : `int`
            the cluster index of the above cell
        labels : `array`
            a 1d-array containing the root of each index

        Returns
        -------
        labels : `array`
            the updated labels array where the left cluster's root has been updated to the above
            cluster's root
    """
    labels[cell_l] = find(cell_a, labels)
    return labels


def match_labels(grid, labels, dim):
    """
        Matches all cluster labels to their root grid. Improves cluster processing

        Parameters
        ----------
        grid : `array`
            the 2d-array containing cluster labels
        labels : `array`
            a 1d-array containing the root of each cluster index
        dim : `int`
            the dimension of the grid

        Returns
        -------
        grid : `array`
            the grid, updated so all cells in a cluster show the root for easy processing
    """
    for row in range(dim):
        for col in range(dim):
            grid[row][col] = find(grid[row][col], labels)
    return grid


def find_centre(cl_idx, grid, dim):
    """
        Locates all cells within a cluster and then finds centre using mean.
        Requires clusters to have been matched using `match_labels(...)`

        Parameters
        ----------
        cl_idx : `int`
            the root index of the cluster of interest
        grid : `array`
            the 2d-array of cluster labels
        dim : `int`
            the dimension of the grid

        Returns
        -------
        centre : `array`
            a 1d-array of form `[x_avg, y_avg]` pointing to the centre of the cluster
    """
    cells = []
    for row in range(dim):
        for col in range(dim):
            if grid[row][col] == cl_idx:
                cells.append([col, row])
    centre = np.mean(cells, axis=0)
    return centre


def locate_centres(cl_inds, grid, dim):
    """
        Locates centres of all clusters in grid using `find_centre(...)`

        Parameters
        ----------
        cl_inds : `array`
            an array containing the root clutster index of all clusters.
            Should not include `0`
        grid : `array`
            the 2d-array of cluster labels
        dim : `int`
            the dimension of the grid

        Returns
        -------
        centres : `array`
           a 2d-array of the locations of all cluster centres of form `[[x1, y1], [x2, y2], ...]`
    """
    centres = []
    for cl_idx in cl_inds:
        centres.append(find_centre(cl_idx, grid, dim))
    return np.array(centres)


tlabel, tlabels = HK_cluster(test_grid, dim)

# print(np.unique(tlabel)[1:])
tlabel = match_labels(tlabel, tlabels, dim)
plt.imshow(tlabel)
plt.colorbar()

clusters = np.unique(tlabel)[1:]
centres = locate_centres(clusters, tlabel, dim)

plt.scatter(*centres.T, marker="o", color="red")
# %%
