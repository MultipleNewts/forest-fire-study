# %%
import numpy as np
import matplotlib.pyplot as plt


# ### TESTING VARS ### #
dim = 10
test_grid = np.random.choice([0, 1, 2], size=(dim, dim), p=[0.7, 0.29, 0.01])
# #################### #


def HK_cluster(grid, dim):
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
        label : `array`
            a 2d-array containing labelled clusters (Note: clusters are matched through roots,
            so they may not directly show the correct number in this array.
            Run `match_labels(...)` to fix).
        labels : `array`
            a 1d-array containing the root of each index. Reduces time waste from doubling-back.
    """
    largest_label = 0
    label = np.zeros((dim, dim), dtype=int)
    labels = np.arange(dim*dim)

    # perform raster scan
    for row in range(dim):
        for col in range(dim):
            if grid[row][col] == 1:
                try:
                    left = label[row-1][col]
                except TypeError:
                    left = 0
                try:
                    above = label[row][col-1]
                except TypeError:
                    above = 0
                if (left == 0) and (above == 0):
                    largest_label += 1
                    label[row][col] = largest_label
                elif (left != 0) and (above == 0):
                    label[row][col] = find(left, labels)
                elif (left == 0) and (above != 0):
                    label[row][col] = find(above, labels)
                else:
                    labels = union(left, above, labels)
                    label[row][col] = find(left, labels)

    return label, labels[:largest_label+1]


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


def match_labels(label, labels, dim):
    """
        Matches all cluster labels to their root label. Improves cluster processing

        Parameters
        ----------
        label : `array`
            the 2d-array containing cluster labels
        labels : `array`
            a 1d-array containing the root of each cluster index
        dim : `int`
            the dimension of the grid

        Returns
        -------
        label : `array`
            the label grid, updated so all cells in a cluster show the root for easy processing
    """
    for row in range(dim):
        for col in range(dim):
            label[row][col] = find(label[row][col], labels)
    return label


tlabel, tlabels = HK_cluster(test_grid, dim)

tlabel = match_labels(tlabel, tlabels, dim)
plt.imshow(tlabel)
plt.colorbar()
# %%
