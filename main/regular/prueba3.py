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
    X_train, X_te, y_train, y_test = train_test_split(X, y, test_size=0.33)

    ###################################
    #####      NORMALIZATION      #####
    ###################################
    # sc = StandardScaler()
    # X_tr_std = sc.fit_transform(X_tr)
    # X_te_std = sc.transform(X_te)

    #########################################
    #####      DATA TRANSFORMATION      #####
    #########################################
    # pca = PCA(n_components=2)
    #
    # X_train = pca.fit_transform(X_tr_std)
    # X_test = pca.transform(X_te_std)

    # X_train = X_tr
    # X_test = X_te

    # X_train = X_train[:,4:]
    # X_test = X_test[:,4:]

    #########################################
    #####      DATA CLASSIFICATION      #####
    #########################################

    print(X, y)

    clf = Alfredo(n_trees=1, perc=0.75, bagg=True)
    clf10 = Alfredo(n_trees=10, perc=0.75, bagg=True)
    clf100 = Alfredo(n_trees=100, perc=0.75, bagg=True)
    RF = RandomForestClassifier(n_estimators=100)

    clf.fit(X_train, y_train)
    clf10.fit(X_train, y_train)
    clf100.fit(X_train, y_train)
    RF.fit(X_train, y_train)

    print(1 - clf100.score(X_te, y_test))
    print(1 - RF.score(X_te, y_test))


if __name__ == "__main__":
    main()
