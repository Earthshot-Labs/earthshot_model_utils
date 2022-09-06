import numpy as np

def chapman_richards_set_ymax(x, k, p):
    """
    Chapman-Richards formula with y_max parameter specified in the array of independent variables x
    to remove this variable from estimation in the curve fitting procedure.

    Parameters
    ----------
    x : n by 2 array of independent variables. First column is time in years, second column is
        constant value of ymax for the site for all years.
    k : [float]
        k parameter
    p : [float]
        p parameter

    Returns
    -------
    vector of y values
    """
    y = x[: ,1] * np.power( (1 - np.exp(-k * x[: ,0])), p)
    return y

def chapman_richards_set_ymax_and_p(x, k):
    """
    Chapman-Richards formula with y_max and p parameters specified in the array of independent variables x
    to remove them from estimation in the curve fitting procedure.

    Parameters
    ----------
    x : n by 2 array of independent variables. First column is time in years, second column is
        constant value of ymax for the site for all years, third column is constant value of p
        for all years.
    k : [float]
        k parameter

    Returns
    -------
    vector of y values
    """
    y = x[: ,1] * np.power( (1 - np.exp(-k * x[: ,0])), x[: ,2])
    return y