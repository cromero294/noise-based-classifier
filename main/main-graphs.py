import sys

sys.path.append('/home/cromero/noise-based-classifier/')

import resources.properties as properties
from resources.PlotModel import *
import datasets.DatasetGenerator as data

from src.ClasificadorRuido import *


def main():
    model = "waveform.csv"

    data = np.load("../data/" + model + "_data.npy")
    rfscore = np.load("../data/" + model + "_data_random-forest.npy")

    print(data.shape)

    for i in range(0, 100, 10):
        plt.plot(np.arange(0.01, 0.99, 0.01), data.mean(axis=0)[:, i], linestyle='-')

    plt.plot(np.arange(0.01, 0.99, 0.01), data.mean(axis=0)[:, -1], linestyle='-')

    plt.axhline(y=rfscore.mean(), color='m', linestyle='-')

    legend = list(map(lambda x: str(x), range(1, 111, 10)))
    legend.append('RF')

    plt.legend(legend, loc='upper right')

    plt.title("Sine dataset")
    plt.ylabel("Err")
    plt.xlabel("Data randomization")

    plt.ylim(0.04, 0.12)

    plt.grid()

    plt.tight_layout()
    # plt.show()
    plt.savefig("../plots/err-random-"+model+"-several_n_trees.png")

if __name__ == "__main__":
    main()
