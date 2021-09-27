import numpy as np
from scipy import stats
from llcf_test.zeta import zetas
from llcf_test.util import normalize


def conf_ints(X, Y, autoscale="minmax", bootstrap_iters=100, alpha=0.05, gamma=0.0):
    """
    Construct a confidence interval of the gamma-trimmed mean of indicator functions. If 1 lies inside the confidence
    interval, then one can accept the null-hypothesis at alpha significance level.

    Null-hypothesis: There exists a locally Lipschitz continuous function from X to Y with probability 1-gamma.

    The null-hypothesis is tested in both directions, from X to Y, and from Y to X.

    Since k-NN search can be affected quite heavily by different scales in the variables, scaling should be done as part
    of preprocessing. There are two built-in autoscaling option: min-max scaling (which runs by default), or rank
    normalization (or None to skip autoscaling). It should be noted that min-max scaling keeps the local Lipschitz
    property, but it's only a linear transform. On the other hand, rank normalization is a non-linear transform, but it
    keeps the local Lipschitz property only asymptotically, so if the sample size is not large it may introduce issues.

    :param X: Sample points of the random vector X
    :type X: numpy.ndarray
    :param Y: Sample points of the random vector Y
    :type Y: numpy.ndarray
    :param autoscale: How X and Y should be autoscaled. Can be either 'minmax' (default), 'rank' or None.
    :type autoscale: str
    :param bootstrap_iters: Number of bootstrap iterations to use for quantile calculation. 100 by default.
    :type bootstrap_iters: int
    :param alpha: The expected significance level. 0.05 by default.
    :type alpha: float
    :param gamma: The trimming parameter (two-sided, gamma/2 is trimmed on each end of the distribution). 0 by default.
    :type gamma: float
    :return: confidence interval from X to Y, and from Y to X
    :rtype: tuple, tuple
    """
    if X.shape[0] != Y.shape[0]:
        raise Exception("Inputs X and Y don't have the same number of rows!")

    X_prep, Y_prep = normalize(X, Y, autoscale)

    zeta_X, zeta_Y, k = zetas(X_prep, Y_prep)

    prob_X = np.sum(zeta_X == 1.0) / zeta_X.shape[0]
    prob_Y = np.sum(zeta_Y == 1.0) / zeta_Y.shape[0]

    p_zeta_eq_1 = np.zeros((bootstrap_iters, 2))

    for i in range(bootstrap_iters):
        indices = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)
        unique_indices, counts = np.unique(indices, return_counts=True)

        X_boot = X_prep[unique_indices, :]
        Y_boot = Y_prep[unique_indices, :]

        zeta_X, zeta_Y, k = zetas(X_boot, Y_boot, autoscale=None)

        p_zeta_eq_1[i, 0] = stats.trim_mean(np.repeat(zeta_X, counts) == 1.0, gamma) - prob_X
        p_zeta_eq_1[i, 1] = stats.trim_mean(np.repeat(zeta_Y, counts) == 1.0, gamma) - prob_Y

    return (prob_X + np.quantile(p_zeta_eq_1[:, 0], alpha), prob_X + np.quantile(p_zeta_eq_1[:, 0], 1-alpha)),\
           (prob_Y + np.quantile(p_zeta_eq_1[:, 1], alpha), prob_Y + np.quantile(p_zeta_eq_1[:, 1], 1-alpha))
