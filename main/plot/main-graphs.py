import sys

sys.path.append('/home/cromero/noise-based-classifier/')

import resources.properties as properties
from resources.PlotModel import *

from src.ClasificadorRuido import *


def main():
    model = "wdbc.csv"

    DIR = properties.SYNTHETIC

    if model.find(".") > -1:
        DIR = properties.DATASETS

    data = np.load(properties.DATA + DIR + model + "_data_ALFREDO.npy")
    rfscore = np.load(properties.DATA + DIR + model + "_data_random-forest_ALFREDO.npy")

    print(data.shape)

    model_title = str.upper(model[0]) + model[1:]
    plt.figure(figsize=(20, 10))

    """
    ----------------------------------------
                   FIRST PLOT
    ----------------------------------------
    """

    plt.subplot(1, 2, 1)

    plt.title(model_title + " perc. random")

    lst = [0, 4, 9, 49, 99]

    for i in lst:
        plt.plot(np.arange(0.01, 0.99, 0.01), data.mean(axis=0)[:, i], linestyle='-')

    plt.axhline(y=rfscore.mean(), color='m', linestyle='-')

    legend = list(map(lambda x: str(x+1), [0, 4, 9, 49, 99]))
    legend.append('RF')

    plt.legend(legend, loc='upper right', ncol=2)

    plt.ylabel("Err")
    plt.xlabel("Data randomization")

    plt.ylim()

    plt.grid()

    """
    ----------------------------------------
                   SECOND PLOT
    ----------------------------------------
    """

    plt.subplot(1, 2, 2)

    plt.title(model_title + " n. trees - err.")

    lst = [0, 4, 9, 48, 97]

    print(data.mean(axis=0).shape)

    for i in lst:
        plt.plot(range(1, 101, 1), data.mean(axis=0)[i, :], linestyle='-')

    plt.axhline(y=rfscore.mean(), color='m', linestyle='-')

    legend = list(map(lambda x: str(x / 100), [1, 5, 10, 50, 99]))
    legend.append('RF')

    plt.legend(legend, loc='upper right', ncol=2)

    plt.ylabel("Err")
    plt.xlabel("N. trees")

    plt.grid()

    plt.tight_layout()

    # plt.show()
    plt.savefig(properties.PLOTS + "ALFREDO/PNG/2-plots_" + model + ".png")
    plt.savefig(properties.PLOTS + "ALFREDO/EPS/2-plots_" + model + ".eps")


if __name__ == "__main__":
    main()
