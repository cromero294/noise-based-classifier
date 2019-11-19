import sys

sys.path.append('/home/cromero/noise-based-classifier/')

import datasets.DatasetGenerator as data
from resources.PlotModel import *

from src.ClasificadorRuido import *
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm


def main():
    model = "sine"

    data = np.load("../data/" + model + "_data.npy")
    rfscore = np.load("../data/" + model + "_data_random-forest.npy")

    # print(data.mean(axis=0)[:, -1][::10])

    # print(data)
    # print(data.shape)
    # print()
    # print(data.mean(axis=0))
    # print(data.mean(axis=0)[:, -1])
    # print(data.mean(axis=0).shape)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=False)

    axs[0].plot(np.arange(0.01, 0.99, 0.01), data.mean(axis=0)[:, -1], linestyle='-')
    axs[0].axhline(y=rfscore.mean(), color='m', linestyle='-')

    for i in range(1, 98, 10):
        axs[1].plot(range(1, 101, 1), data.mean(axis=0)[i, :], linestyle='-.')

    axs[1].legend(list(map(lambda x: str(x/100), range(1, 98, 10))), loc='upper right')

    fig.suptitle(model + ' plotting')

    axs[0].grid()
    axs[1].grid()

    plt.ylim(0.06, 0.1)

    plt.show()

    # fig.show()


if __name__ == "__main__":
    main()
