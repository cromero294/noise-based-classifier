# sys.path.append('/home/cromero/noise-based-classifier/')

import datasets.DatasetGenerator as data
from resources.PlotModel import *

from src.ClasificadorRuido import *
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm


def main():
    model = "sine"

    data = np.load("../data/" + model + "_data.npy")

    print(data)
    print(data.shape)
    print()
    print(data.mean(axis=0))
    print(data.mean(axis=0)[:, -1])
    print(data.mean(axis=0).shape)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=False)

    axs[0].plot(np.arange(0.05, 0.975, 0.025), data.mean(axis=0)[:, -1], linestyle='-.')
    for i in range(1, 37, 5):
        axs[1].plot(range(1, 101, 1), data.mean(axis=0)[i, :], linestyle='-.')

    plt.ylim(0.02, 0.06)

    plt.show()

    # fig.suptitle('Threenorm Plotting')
    # fig.show()


if __name__ == "__main__":
    main()
