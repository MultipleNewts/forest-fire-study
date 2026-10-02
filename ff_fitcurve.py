# %%
from scipy.optimize import curve_fit


# %%
def powerlaw(x, a, k, b):
    return a*(x**k) + b


def power_fit(x_data, y_data):
    a, k, b = curve_fit(powerlaw, x_data, y_data, maxfev=2000)[0]  # , p0=[60000, -1, 3000])[0]
    return (a * (x_data**k) + b), (a, k, b)

# %%
