# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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

    for col in grid[0]:
        if ref_grid[0][col] == 1:
            slf = grid[0][col]
            above = grid[dim-1][col]
            if above != 0:
                union(above, slf, labels)

    for row in range(dim):
        if ref_grid[row][0] == 1:
            slf = grid[row][0]
            left = grid[row][dim-1]
            if left != 0:
                union(left, slf, labels)

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


def union(cell_upd, cell_og, labels):
    """
        Updates the root of a cluster to bind two clusters

        Parameters
        ----------
        cell_upd : `int`
            the cluster index of the cell to be propagated (typically above cell)
        cell_og : `int`
            the cluster index of the cell to be updated (typically left cell)
        labels : `array`
            a 1d-array containing the root of each index

        Returns
        -------
        labels : `array`
            the updated labels array where the (upd) cluster's root has been updated to the (og)
            cluster's root
    """
    labels[cell_upd] = find(cell_og, labels)
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


def pbc_correct(cells, dim):
    """
        Any clusters that encounter the PBCs (i.e. go over the boundary)
        and do not spread across the whole grid are suitably translated.

        Parameters
        ----------
        cells : `array`
            the 2d-array of cell coordinates for the cluster
        dim : `int`
            the dimension of the grid

        Returns
        -------
        cells : `array`
            a 2d-array of form `[[x1, y1], [x2,y2], ...]` containing the PDC-corrected
            coordinates of all cells in the cluster
    """
    cells = np.array(cells)
    # find unqiue row and column coords
    cols = np.unique(cells[:, 0])
    rows = np.unique(cells[:, 1])
    # if the cluster features cells in the first and last columns
    if np.isin([0, (dim-1)], cols).all():
        # and does NOT stretch the whole grid
        if not np.array_equal(cols, np.arange(dim)):
            # the first gap in the cluster is located
            for i in range(dim):
                if i not in cols:
                    offset = i
            # and all column values past this gap are shifted down
            cells[np.where(cells[:, 0] > offset), 0] -= dim
    # repeat for rows
    if np.isin([0, (dim-1)], rows).all():
        if not np.array_equal(rows, np.arange(dim)):
            for i in range(dim):
                if i not in rows:
                    offset = i
            cells[np.where(cells[:, 1] > offset), 1] -= dim
    return cells


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
        cells : `array`
            a 2d-array of form `[[x1, y1], [x2,y2], ...]` containing the coordinates
            of all cells in the cluster
    """
    cells = []
    for row in range(dim):
        for col in range(dim):
            if grid[row][col] == cl_idx:
                cells.append([col, row])
    # PBC correct cells
    cells = pbc_correct(cells, dim)
    centre = np.mean(cells, axis=0)
    # if the mean is negative (due to PBC corrections) shift it to the correct side of the grid
    centre = np.where(centre >= 0, centre, centre + dim - 1)
    return centre, cells


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
        clusters : `array`
            a 3d-array of all the cells coordinates in all clusters
            of form `[cluster no.][cell number][0:x||1:y]`
    """
    centres = []
    clusters = []
    for cl_idx in cl_inds:
        centre, cluster = find_centre(cl_idx, grid, dim)
        centres.append(centre)
        clusters.append(cluster)
    return (np.array(centres), clusters)


def compute_radius(cells, centre):
    """
        Computes the square-root of the mean square distance of each cell from the centre,
        providing a value of the "radius" of the cluster.

        Parameters
        ----------
        cells : `array`
            a 2d-array containing the coordinate positions of all cells within the cluster
        centre : `array`
            a 1d-array contianing the coordinates of the cluster centre, of form `[x, y]`

        Returns
        -------
        RMS_radius : `float`
            the RMS "radius" of the cluster
    """
    sq_diff = 0
    N = len(cells)
    for cell in cells:
        vdiff = cell - centre
        sq_diff += np.sum((vdiff)**2)
    RMS_radius = np.sqrt(sq_diff/N)
    return RMS_radius


# for debugging and testing. Also shows running order
if __name__ == "__main__":
    # ### TESTING VARS ### #
    dim = 25
    test_grid = np.random.choice([0, 1, 2], size=(dim, dim), p=[0.5, 0.49, 0.01])
    # #################### #

    # run HK clustering algorithm to find clusters
    tlabel, tlabels = HK_cluster(test_grid, dim)

    # match all cluster labels to root
    tlabel = match_labels(tlabel, tlabels, dim)

    # visualisation for debugging
    plt.imshow(tlabel)
    plt.colorbar()

    # finds all cluster indecies
    clusters = np.unique(tlabel)[1:]
    # locates all cluster centres and the cells within each cluster
    centres, clusters = locate_centres(clusters, tlabel, dim)

    # plots the centres of each cluster for debugging
    plt.scatter(*centres.T, marker="o", color="red")

    # computes the "radius" of each cluster
    radii = []
    for i, centre in enumerate(centres):
        radii.append(compute_radius(clusters[i], centre))

# %%
