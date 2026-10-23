# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
def model(x, k, a, c):
    """
        Power Model of form `k*(x**a)+c`

        Parameters
        ----------
        x : `array`
            an array of x values
        k : `float`
            the value of the coefficient in the model
        a : `float`
            the power of `x` in the model
        c : `float`
            the offset value in the model

        Returns
        -------
        model : `array`
            the `x` values transformed according to the model
    """
    return k*(np.power(x, a))+c


def MSE(model, data):
    """
        the mean-square-error between a model and a set of data

        Parameters
        ----------
        model : `array`
            the model-predicted data
        data : `array`
            the observed data

        Returns
        -------
        mse : `float`
            the mean-square-error between the observed and predicted data
    """
    return np.mean(np.power((data-model), 2))


def fit_params(data, x, ntests=10, kran=[1, 7], pran=[-1, 0], oran=[3.5, 4.0]):
    """
        Fit parameters of a power model to a set of data

        Parameters
        ----------
        data : `array`
            the observed data
        x : `array`
            an array of x values
        ntests : `int`
            the number of values to include within each parameter range
        kran : `array`
            the range of coefficient (`k`) values to test in form `[min, max]`
        pran : `array`
            the range of power (`a`) values to test in form `[min, max]`
        oran : `array`
            the range of offset (`c`) value to test in form `[min, max]`

        Returns
        -------
        params : `tuple`
            the set of parameters that minimise the MSE function, in form `(k, a, c)`
    """
    epsilon = 1e-12
    ptests = np.linspace(pran[0], pran[1], ntests)+epsilon
    otests = np.linspace(oran[0], oran[1], ntests)+epsilon
    ktests = np.linspace(kran[0], kran[1], ntests)+epsilon
    # mses[p_ind][o_ind][k_ind]
    mses = []
    for p in ptests:
        each_p = []
        for o in otests:
            each_o = []
            for k in ktests:
                each_o.append(MSE(model(x, k, p, o), data))
            each_p.append(each_o)
        mses.append(each_p)
    mses = np.array(mses)
    min = np.unravel_index(np.argmin(mses), mses.shape)
    return (ktests[min[2]], ptests[min[0]], otests[min[1]])


def plot_dat_mod(x, data, params, data_label="Data", model_label="Model", y_label="Value"):
    """
        plots a set of observed data with a model

        Parameters
        ----------
        x : `array`
            the array of x data
        data : `array`
            the observed data
        model : `array`
            the model-predicted data
        data_label : `str`
            the label for the data, displayed on the legend
        model_label : `str`
            the label for the model, displayed on the legend
        y_label : `str`
            the label for the y-axis
    """
    plt.scatter(x, data, color="blue",
                linestyle="-",
                linewidth=0.8,
                zorder=2,
                label=data_label,
                marker="x")
    x_smooth = np.linspace(np.min(x), np.max(x), len(x)*10)
    plt.plot(x_smooth,
             model(x_smooth, *params),
             color="red",
             linestyle="-",
             linewidth=1.8,
             zorder=2,
             label=model_label)
    plt.title("Power Model Fit on Average Cluster Size")
    plt.xlabel("Time Step")
    plt.ylabel(y_label)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.show()

# %%
