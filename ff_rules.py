# %%
import numpy as np


def pbc(index, dim=100):
    """Enforces the Periodic Boundary Condition

        Parameters
        ----------
        index : `int`
            the index to test against the pbc
        dim : `int`
            the dimension of the board, default `dim=100`. ensure pbc works correctly

        Returns
        -------
        index : `int`
            the PBC-corrected index
        """
    if index < 0:
        return (dim-1)
    if index >= dim:
        return 0
    return index


def check_neighbours(board, coord_x, coord_y, dim=100):
    """Checks the state of all neighbouring cells on the board

        Parameters
        ----------
        board : `array`
            the 2d array containing the board states
        coord_x : `int`
            the x/column coordinate of the cell of interest
        coord_y : `int`
            the y/row coordinate of the cell of interest
        dim : `int`
            the dimension of the board, default `dim=100`. ensure pbc works correctly

        Returns
        -------
        neighbour_states : `array`
            a 1d array of the states of the cell of interest's neighbours.
            The order is left, right, above, below
        """
    neighbours = [
            board[coord_y, pbc(coord_x-1, dim)],
            board[coord_y, pbc(coord_x+1, dim)],
            board[pbc(coord_y-1, dim), coord_x],
            board[pbc(coord_y+1, dim), coord_x],
        ]
    return neighbours


def update_cell(board, coord_x, coord_y, dim=100, p=0.1, fpratio=0.1):
    """Updates a cell based on a set of rules

        Parameters
        ----------
        board : `array`
            the 2d array containing the board states
        coord_x : `int`
            the x/column coordinate of the cell of interest
        coord_y : `int`
            the y/row coordinate of the cell of interest
        dim : `int`
            the dimension of the board, default `dim=100`. ensure pbc works correctly

        Returns
        -------
        new_state : `int`
            the new state of the cell according to the rules
        """
    state = board[coord_y, coord_x]
    neighbours = check_neighbours(board, coord_x, coord_y, dim)
    # ########### RULES ############ #
    # if a cell is a tree and a neighbour is burning, it burns
    if (state == 1) and (2 in neighbours):
        return 2
    # if a cell was burning, it is now empty
    elif state == 2:
        return 0
    # if a cell is empty, it may become a tree with probability p
    elif state == 0:
        return np.random.choice([0, 1], p=[1-p, p])
    # if a cell is a tree, it may start burning with probability f=p*fpratio
    elif state == 1:
        f = p * fpratio
        return np.random.choice([1, 2], p=[1-f, f])
    # in case an unexpected outcome occurs, raise an error
    else:
        raise ValueError("Unaccounted Outcome")
    # ############################## #

# %%
