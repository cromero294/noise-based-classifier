import sys

sys.path.append('/home/cromero/noise-based-classifier/')

import resources.properties as properties
from resources.PlotModel import *
import datasets.DatasetGenerator as data

from src.ClasificadorRuido import *


def main():
    model = "ringnorm"

    data = np.load("../data/" + model + "_data_ALFREDO.npy")
    rfscore = np.load("../data/" + model + "_data_random-forest_ALFREDO.npy")

    print(data.shape)

    model_title = str.upper(model[0]) + model[1:]

    plt.subplot(1, 2, 1)

    plt.title(model_title + " perc. random")

    lst = [0, 4, 9, 49, 99]

    for i in lst:
        plt.plot(np.arange(0.01, 0.99, 0.01), data.mean(axis=0)[:, i], linestyle='-')

    plt.axhline(y=rfscore.mean(), color='m', linestyle='-')

    legend = list(map(lambda x: str(x+1), [0, 4, 9, 49, 99]))
    legend.append('RF')

    plt.legend(legend, loc='upper right')

    plt.ylabel("Err")
    plt.xlabel("Data randomization")

    plt.ylim()

    plt.grid()

    ########################################################################################

    plt.subplot(1, 2, 2)

    plt.title(model_title + " n. trees - err.")

    lst = [0, 4, 9, 48, 97]

    print(data.mean(axis=0).shape)

    for i in lst:
        plt.plot(range(1, 101, 1), data.mean(axis=0)[i, :], linestyle='-')

    plt.axhline(y=rfscore.mean(), color='m', linestyle='-')

    legend = list(map(lambda x: str(x / 100), [1, 5, 10, 50, 99]))
    legend.append('RF')

    plt.legend(legend, loc='upper right')

    plt.ylabel("Err")
    plt.xlabel("N. trees")

    plt.grid()

    plt.tight_layout()
    plt.show()
    # plt.savefig("../plots/several-trees_err-random/err-random-"+model+"-several_n_trees.png")


if __name__ == "__main__":
    main()
