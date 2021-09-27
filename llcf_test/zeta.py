import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from llcf_test.util import normalize


def zetas(X, Y, k=None, autoscale="minmax", verbose=False):
    """
    Calculate the local zeta estimators in both directions.

    :param X: Sample points of the random vector X
    :type X: numpy.ndarray
    :param Y: Sample points of the random vector Y
    :type Y: numpy.ndarray
    :param k: Neighbourhood size to use for the estimator. Ceil(sqrt(n)) by default.
    :type k: int
    :param autoscale: How X and Y should be autoscaled. Can be either 'minmax' (default), 'rank' or None.
    :type autoscale: str
    :param verbose: Indicates whether progress information should be printed. False by default.
    :type verbose: bool
    :return: zeta estimators for each point in X and Y, and k
    :rtype: numpy.ndarray, numpy.ndarray, int
    """
    if X.shape[0] != Y.shape[0]:
        raise Exception("Inputs X and Y don't have the same number of rows!")

    X_prep, Y_prep = normalize(X, Y, autoscale)

    if k is None:
        k = int(np.ceil(np.sqrt(X.shape[0])))

    if verbose:
        print("Calculating k neighbourhoods in J...")

    J_prep = np.concatenate((X_prep, Y_prep), axis=1)
    tree = cKDTree(J_prep)
    _, nn_J = tree.query(J_prep, k=k)

    if verbose:
        print("Calculating zetas from X to Y...")

    zeta_X = __local_zetas(X_prep, nn_J, k, verbose)

    if verbose:
        print("Calculating zetas from Y to X...")

    zeta_Y = __local_zetas(Y_prep, nn_J, k, verbose)

    return zeta_X, zeta_Y, k


def __local_zetas(X_prep, nn_J, k, verbose=False):
    zeta_X = np.zeros(X_prep.shape[0])

    for i in range(X_prep.shape[0]):
        if verbose and i % 10 == 0:
            print(i / X_prep.shape[0])

        if X_prep.shape[1] == 1:
            x_min = np.min(X_prep[nn_J[i, :], 0])
            x_max = np.max(X_prep[nn_J[i, :], 0])
            zeta_X[i] = k / np.sum(np.logical_and(x_min <= X_prep, X_prep <= x_max))
        else:
            try:
                cvxh = ConvexHull(X_prep[nn_J[i, :], :], incremental=False)
                A = np.dot(X_prep, cvxh.equations[:, :-1].T) + cvxh.equations[:, -1].reshape(1, -1)
                zeta_X[i] = k / np.sum(np.sum(A <= 1e-10, axis=1) == cvxh.equations.shape[0])
            except Exception:
                zeta_X[i] = np.nan

    return zeta_X
