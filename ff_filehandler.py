# %%
import numpy as np


# %%
# writes sim data to a file
def write_data(data, dim, dur, name):
    # uses custom .fdat extension for storage
    filename = f"{name}.fdat"
    with open(filename, "w") as file:
        # appends the grid dims and sim duration to start of file
        file.write(f"{dim},{dur},")
        # flattens data
        temp = np.array(data).ravel()
        # converts flattened data to csv string
        temp1 = np.array2string(temp, separator=",", threshold=np.inf)[1:-1]
        file.write("".join(temp1))


# reads stored data from files
def read_data(name):
    # finds correct .fdat file
    filename = f"{name}.fdat"
    with open(filename) as file:
        contents = file.read()
        # uses list comprehension to convert csv back into a list of ints
        csv = [int(x) for x in contents.split(",")]
        # obtains dim and dur from file header
        dim = csv[0]
        dur = csv[1]
        # separates actual data from header
        data = np.array(csv[2:])
        ndata = []
        # splits data into the correct number of boards
        boards = np.split(data, dur)
        for board in boards:
            # reshapes boards into original square shape
            temp = board.reshape((dim, dim))
            ndata.append(temp)
    return ndata


# manages read data to provide same outputs as normal running
def use_stored(name="test"):
    ndata = read_data(name)
    tdata = []
    for board in ndata:
        # counts tile states to provide tile data
        temp = board.ravel()
        tdata.append(np.bincount(temp, minlength=3))
    return tdata, ndata
