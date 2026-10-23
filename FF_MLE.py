# %%
from scipy.special import zeta
from scipy.differentiate import derivative
from scipy.optimize import fsolve
import numpy as np


# %%
def riemann_zeta(s):
    return zeta(s)


def zeta_nprime(at_val, n=1):
    return derivative(riemann_zeta, at_val, order=n).df


def zeta_ratio(s):
    z_prime = zeta_nprime(s)
    z = riemann_zeta(s)
    # print(z_prime/z)
    return np.abs(z_prime/z)


def check_range(val, range):
    if (val >= range[0] and val <= range[1]):
        return True
    return False


def equality_func(x, const):
    return zeta_ratio(x)-const


def find_param(fixed_val, strt=5):
    alpha = fsolve(equality_func, strt, args=fixed_val, maxfev=800)
    return alpha[0]


def find_MLE(check_val, strt=0, step=500, niter=500, acc=1e-6):
    """
        **!!OBSOLETE!!**\n
        Finds the MLE parameter by evaluating expression.
        Uses a binary search to locate best parameter.

        Parameters
        ----------
        check_val : `float`
            the target value to optimise the expression towards
        start : `float`
            the smallest value possible in the search, result will be `>strt`
        step : `float`
            the maximum possible value in the search, result will be `<=step`
        niter : `int`
            the maximum number of iterations that will be run,
            if this is reached the current value will be output
        acc : `float`
            the accuracy level to get the result to,
            if the value gets within this range of the reference value the value is output

        Returns
        -------
        value : `float`
            the "alpha" value provided to the function that provides the
            closest outcome to the `check_val`
    """
    check_val = np.abs(check_val)
    ref = [check_val-acc, check_val+acc]
    print(ref)
    cur = strt
    direction = -1
    for i in range(niter):
        cur += step
        print(zeta_ratio(cur))
        if check_range(zeta_ratio(cur), ref) is False:
            step *= 0.5
            cur += direction*step
            if zeta_ratio(cur) > check_val:
                direction = -1
            else:
                direction = 1
        else:
            break
    return cur


def get_error(val, N):
    """
        Finds the error associated with the input parameter value.
        Formula uses the computed Fisher Information.

        Parameters
        ----------
        val : `float`
            the value to find the error for
        N : `int`
            the size of the data set, used to scale

        Returns
        -------
        error : `float`
            the standard error on the input val
    """
    term1 = zeta_nprime(val, n=2)/riemann_zeta(val)
    term2 = np.power((zeta_nprime(val, n=1)/riemann_zeta(val)), 2)
    comb_term = np.sqrt((term2-term1))
    scale = 1/np.sqrt(N)
    return scale/comb_term
