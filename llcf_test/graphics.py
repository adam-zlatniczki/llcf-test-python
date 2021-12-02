from matplotlib import pyplot as plt
from llcf_test.zeta import zetas
import numpy as np


def __add_zeta_histograms_to_axes(X, Y, axis1, axis2, autoscale="minmax", max_bins=15):
    zeta_X, zeta_Y, _ = zetas(X, Y, autoscale=autoscale)

    axis1.set_title("Zeta distribution (X to Y)")
    axis1.set_xlim(left=0.0, right=1.1)

    axis2.set_title("Zeta distribution (Y to X)")
    axis2.set_xlim(left=0.0, right=1.1)

    # plot local zeta histogram for X
    values, counts = np.unique(zeta_X, return_counts=True)

    if values.shape[0] <= max_bins:
        if values.shape[0] > 1:
            axis1.bar(values, counts, width=0.8 * np.min(np.diff(values)))
        else:
            axis1.bar(values, counts, width=0.05)
    else:
        axis1.hist(zeta_X, bins=max_bins)

    # plot local zeta histogram for Y
    values, counts = np.unique(zeta_Y, return_counts=True)
    if values.shape[0] <= max_bins:
        if values.shape[0] > 1:
            axis2.bar(values, counts, width=0.8 * np.min(np.diff(values)))
        else:
            axis2.bar(values, counts, width=0.05)
    else:
        axis2.hist(zeta_Y, bins=max_bins)


def plot_zeta_histograms(X, Y, autoscale="minmax", max_bins=15):
    """
    Plots two histograms, indicating the distribution of local zeta estimators in each direction.

    :param X: Sample points of the random vector X
    :type X: numpy.ndarray
    :param Y: Sample points of the random vector Y
    :type Y: numpy.ndarray
    :param autoscale: How X and Y should be autoscaled. Can be either 'minmax' (default), 'rank' or None
    :type autoscale: str
    :param max_bins: Maximal number of bins to use for the histograms
    :type max_bins: int
    :return: Figure
    :rtype: matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2)

    axes[0].set_title("Zeta distribution (X to Y)")
    axes[1].set_title("Zeta distribution (Y to X)")

    __add_zeta_histograms_to_axes(X, Y, axes[0], axes[1], autoscale, max_bins)

    return fig


def scatter_plot_with_zetas(X, Y, autoscale="minmax", max_bins=15):
    if (len(X.shape) > 1 and X.shape[1] > 1) or (len(Y.shape) > 1 and Y.shape[1] > 1):
        raise Exception("Make sure the datasets provided are one-dimensional!")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    axes[0][0].set_title("Joint state space")
    axes[1][0].set_title("Zeta distribution (X to Y)")
    axes[0][1].set_title("Zeta distribution (Y to X)")
    axes[1][1].axis('off')

    axes[0][0].scatter(X, Y)
    __add_zeta_histograms_to_axes(X, Y, axes[1][0], axes[0][1], autoscale, max_bins)

    return fig
