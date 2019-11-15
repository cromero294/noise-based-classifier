import sys

# sys.path.append('/home/cromero/noise-based-classifier/')

import resources.properties as properties
import datasets.DatasetGenerator as data
from resources.PlotModel import *

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

import pandas as pd

from src.ClasificadorRuido import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm


def main():

    X_test, y_test = data.create_dataset(5000, "twonorm")

    print(X_test)

    return

    #########################################
    #####      DATA CLASSIFICATION      #####
    #########################################
    rfclf = RandomForestClassifier(n_estimators=100)

    rfclf.fit(X_train, y_train)
    rfscore = 1 - rfclf.score(X_test, y_test)

    plt.axhline(y=rfscore, color='m', linestyle='--')

    n_trees = 100

    scores = []

    for perc in tqdm(np.arange(0.1, 0.9, 0.01)):
        clf = ClasificadorRuido(n_trees=n_trees, perc=perc)

        score_aux = 0

        for _ in range(100):
            clf.fit(X_train, y_train, random_perc=False)
            score_aux += (1 - clf.score(X_test, y_test))

        scores.append(score_aux / 100)

    plt.plot(np.arange(0.1, 0.9, 0.01), scores, linestyle='-.')

    plt.savefig("../plots/plot_errors_wdbc.png")


if __name__ == "__main__":
    main()
