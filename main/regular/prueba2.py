import sys

sys.path.append('/home/cromero/noise-based-classifier/')

import resources.properties as properties
from resources.PlotModel import *

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from src.ClasificadorRuido import *
from sklearn.ensemble import RandomForestClassifier

from src.Alfredo import *


def get_data(model):
        try:
            dataset = pd.read_csv(properties.DATASET_DIRECTORY + "csv/" + model + ".csv")
            # print(dataset.dtypes)

            # AUXILIAR


            # aux = dataset['class']
            # dataset.drop(labels=['class'], axis=1, inplace = True)
            # dataset.insert(5, 'class', aux)

            cat_columns = ['class']

            if model == "tic-tac-toe":
                cat_columns = dataset.select_dtypes(['object']).columns

            # print(dataset)

            # AUXILIAR

            dataset[cat_columns] = dataset[cat_columns].astype('category')
            dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)
            dataset = dataset.values
            X, y = dataset[:, :-1], dataset[:, -1]

            return X, y
        except IOError:
            print("File \"{}\" does not exist.".format(sys.argv[1]))
            return

def main():

    model = sys.argv[1]

    X, y = get_data(model)

    # X_tr, X_te, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_tr, X_te, y_train, y_test = train_test_split(X, y, test_size=0.1)

    ###################################
    #####      NORMALIZATION      #####
    ###################################
    sc = StandardScaler()
    X_tr_std = sc.fit_transform(X_tr)
    X_te_std = sc.transform(X_te)

    #########################################
    #####      DATA TRANSFORMATION      #####
    #########################################
    pca = PCA(n_components=2)

    X_train = pca.fit_transform(X_tr_std)
    X_test = pca.transform(X_te_std)

    # X_train = X_tr
    # X_test = X_te

    # X_train = X_train[:,4:]
    # X_test = X_test[:,4:]

    #########################################
    #####      DATA CLASSIFICATION      #####
    #########################################

    clf = Alfredo(n_trees=1, perc=0.75, bagg=True)
    clf10 = Alfredo(n_trees=10, perc=0.75, bagg=True)
    clf100 = Alfredo(n_trees=100, perc=0.75, bagg=True)
    clf1000 = Alfredo(n_trees=1000, perc=0.75, bagg=True)

    clf.fit(X_train, y_train)
    clf10.fit(X_train, y_train)
    clf100.fit(X_train, y_train)
    clf1000.fit(X_train, y_train)

    plt.figure(figsize=(12, 3))

    plt.subplot(1, 4, 1)
    plot_model(clf, X_train, y_train, "1 estimators")

    plt.plot()

    plt.subplot(1, 4, 2)
    plot_model(clf10, X_train, y_train, "10 estimators")

    plt.plot()

    plt.subplot(1, 4, 3)
    plot_model(clf100, X_train, y_train, "100 estimators")

    plt.plot()

    plt.subplot(1, 4, 4)
    plot_model(clf1000, X_train, y_train, "1000 estimators")

    plt.plot()

    plt.tight_layout()
    # plt.show()

    plt.savefig(properties.PLOTS + "ALFREDO/PNG/4-plots_" + model + ".png")
    plt.savefig(properties.PLOTS + "ALFREDO/EPS/4-plots_" + model + ".eps")


if __name__ == "__main__":
    main()
