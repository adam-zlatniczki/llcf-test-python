from matplotlib import pyplot as plt
from llcf_test.zeta import zetas


def plot_zeta_histograms(X, Y, autoscale="minmax", bins=10):
    zeta_X, zeta_Y, _ = zetas(X, Y, autoscale=autoscale)

    fig, axes = plt.subplots(1, 2)

    axes[0].set_title("Zeta distribution (X to Y)")
    axes[1].set_title("Zeta distribution (Y to X)")

    axes[0].hist(zeta_X, bins=bins)
    axes[1].hist(zeta_Y, bins=bins)

    return fig
