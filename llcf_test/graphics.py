from matplotlib import pyplot as plt
from llcf_test.zeta import zetas


def plot_zeta_histograms(X, Y, autoscale="minmax", bins=10):
    """
    Plots two histograms, indicating the distribution of local zeta estimators in each direction.

    :param X: Sample points of the random vector X
    :type X: numpy.ndarray
    :param Y: Sample points of the random vector Y
    :type Y: numpy.ndarray
    :param autoscale: How X and Y should be autoscaled. Can be either 'minmax' (default), 'rank' or None
    :type autoscale: str
    :param bins: Number of bins to use for the histograms
    :type bins: int
    :return: Figure
    :rtype: matplotlib.figure.Figure
    """
    zeta_X, zeta_Y, _ = zetas(X, Y, autoscale=autoscale)

    fig, axes = plt.subplots(1, 2)

    axes[0].set_title("Zeta distribution (X to Y)")
    axes[1].set_title("Zeta distribution (Y to X)")

    axes[0].hist(zeta_X, bins=bins)
    axes[1].hist(zeta_Y, bins=bins)

    return fig
