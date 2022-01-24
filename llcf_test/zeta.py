import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from llcf_test.util import normalize


def zetas(X, Y, k=None, autoscale="minmax", vareps=1e-10, p_norm=2, verbose=False):
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
    :param p_norm: The Minkowski p-norm to use in kNN search.
    :type p_norm: int
    :param verbose: Indicates whether progress information should be printed. False by default.
    :type verbose: bool
    :return: zeta estimators for each point in X, Y, and k
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
    _, nn_J = tree.query(J_prep, k=k, p=p_norm, n_jobs=-1)

    if verbose:
        print("Calculating zetas from X to Y...")

    zetas_X = __local_zetas(X_prep, nn_J, k, vareps, verbose)

    if verbose:
        print("Calculating zetas from Y to X...")

    zetas_Y = __local_zetas(Y_prep, nn_J, k, vareps, verbose)

    return zetas_X, zetas_Y, k


def __local_zetas(S_prep, I, k, vareps=1e-10, verbose=False):
    zetas = np.zeros(S_prep.shape[0])

    for i in range(S_prep.shape[0]):
        if verbose and i % 10 == 0:
            print(i / S_prep.shape[0])

        if S_prep.shape[1] == 1:
            s_min = np.min(S_prep[I[i, :], 0])
            s_max = np.max(S_prep[I[i, :], 0])
            zetas[i] = k / np.sum(np.logical_and(s_min <= S_prep, S_prep <= s_max))
        else:
            try:
                cvxh = ConvexHull(S_prep[I[i, :], :], incremental=False)

                # Intentionally ADDING the offset, because QuickHull automatically multiplies it by -1
                E = np.dot(S_prep, cvxh.equations[:, :-1].T) + cvxh.equations[:, -1].reshape(1, -1)

                zetas[i] = k / np.sum(np.sum(E <= vareps, axis=1) == cvxh.equations.shape[0])
            except Exception as e:
                # QuickHull most likely raised an error due to running into a simple linear embedding.
                zetas[i] = np.nan

    return zetas
