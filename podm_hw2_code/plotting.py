from matplotlib import pyplot as plt


def finalize_plot(xlabel: str, ylabel: str, title: str, path=None):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()
