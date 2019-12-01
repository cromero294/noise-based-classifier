import sys

sys.path.append('/home/cromero/noise-based-classifier/')

from resources.PlotModel import *

from src.ClasificadorRuido import *


def main():
    model = "threenorm"

    # data = np.load("../data/" + model + "_data.npy")
    # rfscore = np.load("../data/" + model + "_data_random-forest.npy")

    data = np.load("../data/" + model + "_data_1000.npy")
    rfscore = np.load("../data/" + model + "_data_1000_random-forest.npy")

    fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=False)

    axs[0].plot(np.arange(0.01, 0.99, 0.01), data.mean(axis=0)[:, -1], linestyle='-')
    axs[0].axhline(y=rfscore.mean(), color='m', linestyle='-')

    for i in range(1, 98, 10):
        axs[1].plot(range(1, 1001, 1), data.mean(axis=0)[i, :], linestyle='-.')

    axs[1].axhline(y=rfscore.mean(), color='m', linestyle='-')

    legend = list(map(lambda x: str(x/100), range(1, 98, 10)))
    legend.append('RF')

    axs[1].legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))

    model_title = str.upper(model[0]) + model[1:]
    fig.suptitle(model_title + ' dataset')

    fig.text(0.3, 0., 'random perc.', ha='center')
    fig.text(0.72, 0., 'num. of trees', ha='center')
    fig.text(0.04, 0.5, 'err', va='center', rotation='vertical')

    axs[0].grid()
    axs[1].grid()

    plt.ylim()

    # plt.show()
    plt.savefig("../plots/1000-trees_full-err-random-n_trees/2plot_err-random-n_trees-" + model + ".png")


if __name__ == "__main__":
    main()
