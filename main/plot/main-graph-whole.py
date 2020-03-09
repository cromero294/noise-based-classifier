import sys

sys.path.append('/home/cromero/noise-based-classifier/')

import resources.properties as properties
from resources.PlotModel import *

from src.ClasificadorRuido import *


def main():


    models = ["australian","diabetes","german"]

    zz = 0

    plt.title("Exp. 2 perc. random")
    plt.ylabel("Err")
    plt.xlabel("Data randomization")

    for model in models:

        # DIR = properties.SYNTHETIC
        DIR = properties.DATASETS

        data = np.load(properties.DATA + DIR + model + "_data.npy")
        rfscore = np.load(properties.DATA + DIR + model + "_data_random-forest.npy")
        treescore = np.load(properties.DATA + DIR + model + "_data_tree.npy")
        boostingscore = np.load(properties.DATA + DIR + model + "_data_boosting.npy")
        baggingscore = np.load(properties.DATA + DIR + model + "_data_bagging.npy")

        """
        ----------------------------------------
                       COMPARISON
        ----------------------------------------
        """
        res = data.mean(axis=0)[:, 99]
        print(res)
        print(np.amin(res))
        print(np.where(res == np.amin(res))[0])

        print("MODELO: " + model)
        print()
        print("\t" + properties.COLOR_BLUE + "ALFREDO 50: " + properties.END_C + str(data.mean(axis=0)[49, 99]))
        print("\t" + properties.COLOR_BLUE + "ALFREDO 75: " + properties.END_C + str(data.mean(axis=0)[74, 99]))
        print("\t" + properties.COLOR_BLUE + "RANDOM F.: " + properties.END_C + str(rfscore.mean()))
        print("\t" + properties.COLOR_BLUE + "BOOSTING: " + properties.END_C + str(boostingscore.mean()))
        print("\t" + properties.COLOR_BLUE + "BAGGING: " + properties.END_C + str(baggingscore.mean()))
        print("\t" + properties.COLOR_BLUE + "TREE: " + properties.END_C + str(treescore.mean()))

        # -------------------------------------- #

        model_title = str.upper(model[0]) + model[1:]


        """
        ----------------------------------------
                       FIRST PLOT
        ----------------------------------------
        """

        zz+=1
        plt.subplot(3, 2, zz)

        lst = [0, 4, 9, 49, 99]

        for i in lst:
            plt.plot(np.arange(0.02, 0.99, 0.01), data.mean(axis=0)[1:, i], linestyle='-')

        plt.axhline(y=rfscore.mean(), color='m', linestyle='-')
        plt.axhline(y=boostingscore.mean(), color='g', linestyle='-')
        plt.axhline(y=baggingscore.mean(), color='y', linestyle='-')

        legend = list(map(lambda x: str(x+1), [0, 4, 9, 49, 99]))
        legend.append('RF')
        legend.append('Ada.')
        legend.append('Bagg.')

        plt.legend(legend, loc='upper right', ncol=2)

        plt.ylim()

        plt.grid()

        """
        ----------------------------------------
                       SECOND PLOT
        ----------------------------------------
        """

        zz+=1
        plt.subplot(3, 2, zz)

        # plt.title(model_title + " n. trees - err.")

        lst = [1, 4, 9, 48, 97]

        print(data.mean(axis=0).shape)

        for i in lst:
            plt.plot(range(1, 101, 1), data.mean(axis=0)[i, :], linestyle='-')

        plt.axhline(y=rfscore.mean(), color='m', linestyle='-')
        plt.axhline(y=boostingscore.mean(), color='g', linestyle='-')
        plt.axhline(y=baggingscore.mean(), color='y', linestyle='-')

        legend = list(map(lambda x: str(x / 100), [2, 5, 10, 50, 99]))
        legend.append('RF')
        legend.append('Ada.')
        legend.append('Bagg.')

        plt.legend(legend, loc='upper right', ncol=2)

        plt.grid()

    plt.tight_layout()
    # plt.show()
    plt.savefig(properties.PLOTS + "ALFREDO/PNG/2-plots_" + model + ".png")
    plt.savefig(properties.PLOTS + "ALFREDO/EPS/2-plots_" + model + ".eps")


if __name__ == "__main__":
    main()