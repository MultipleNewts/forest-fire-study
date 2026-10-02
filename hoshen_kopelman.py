# %%
import numpy as np


# test_grid = [
#     [0, 0, 0, 0, 1],
#     [0, 1, 0, 1, 1],
#     [1, 1, 0, 0, 0],
#     [0, 0, 1, 1, 1],
#     [1, 1, 0, 1, 0]
#     ]
dim = 5000
test_grid = np.random.choice([0, 1, 2], size=(dim, dim), p=[0.6, 0.39, 0.01])


def HK_cluster(grid, dim):
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

    return label, labels


def find(cell, labels):
    return labels[cell]


def union(cell1, cell2, labels):
    labels[cell2] = find(cell1, labels)
    return labels


tlabel, tlabels = HK_cluster(test_grid, dim)
print(tlabel)
print(tlabels)
# unioned = np.where(tlabels == 1)[0]
# count = 0
# for val in unioned:
#     count += len(np.where(tlabel == val)[0])
# print(count)
# %%
