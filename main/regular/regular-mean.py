import sys

sys.path.append('/home/cromero/noise-based-classifier/')

import resources.properties as properties
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import datasets.DatasetGenerator as data

from src.ClasificadorRuido import *
from src.Alfredo import *


def get_data():
    if len(sys.argv) > 1:
        try:
            dataset = pd.read_csv(properties.DATASET_DIRECTORY + sys.argv[1])
            print(dataset.dtypes)
            cat_columns = dataset.select_dtypes(['object']).columns

            # AUXILIAR

            # print(dataset)

            # aux = dataset['class']
            # dataset.drop(labels=['class'], axis=1, inplace = True)
            # dataset.insert(13, 'class', aux)
            #
            # print(dataset)
            #
            # cat_columns = ['class']

            # AUXILIAR

            dataset[cat_columns] = dataset[cat_columns].astype('category')
            dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)
            dataset = dataset.values
            X, y = dataset[:,:-1], dataset[:,-1]
        except IOError:
            print("File \"{}\" does not exist.".format(sys.argv[1]))
            return
    else:
        print("File does not exist.")
        return

    return X, y


def main():
    # X, y = get_data()

    model = 'ringnorm'

    X, y = data.create_full_dataset(5300, 20, model)

    y = y.ravel()

    sss = StratifiedShuffleSplit(n_splits=100, test_size=0.2)

    n_estimators = 1000

    tree_clf = tree.DecisionTreeClassifier()
    alf = Alfredo(n_trees=n_estimators, perc=0.75, bagg=True)
    noise_clf = ClasificadorRuido(n_trees=n_estimators, perc=0.5)
    boosting = AdaBoostClassifier(n_estimators=n_estimators)
    bagging = BaggingClassifier(n_estimators=n_estimators)
    rf = RandomForestClassifier(n_estimators=n_estimators)

    tree_scores = []
    alfredo_scores = []
    noise_scores = []
    boosting_scores = []
    bagging_scores = []
    rf_scores = []

    i = 0

    for train_index, test_index in sss.split(X, y):
        print(i)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        tree_clf.fit(X_train, y_train)
        alf.fit(X_train, y_train)
        noise_clf.fit(X_train, y_train)
        boosting.fit(X_train, y_train)
        bagging.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        tree_scores.append(1 - tree_clf.score(X_test, y_test))
        alfredo_scores.append(1 - alf.score(X_test, y_test))
        noise_scores.append(1 - noise_clf.score(X_test, y_test))
        boosting_scores.append(1 - boosting.score(X_test, y_test))
        bagging_scores.append(1 - bagging.score(X_test, y_test))
        rf_scores.append(1 - rf.score(X_test, y_test))

        i += 1

    np.save(properties.DATA + properties.SCORES + model + "_TREE-SCORES", np.array(tree_scores))
    np.save(properties.DATA + properties.SCORES + model + "_ALFREDO-SCORES", np.array(alfredo_scores))
    np.save(properties.DATA + properties.SCORES + model + "_NOISE-SCORES", np.array(noise_scores))
    np.save(properties.DATA + properties.SCORES + model + "_BOOSTING-SCORES", np.array(boosting_scores))
    np.save(properties.DATA + properties.SCORES + model + "_BAGGING-SCORES", np.array(bagging_scores))
    np.save(properties.DATA + properties.SCORES + model + "_RF-SCORES", np.array(rf_scores))


if __name__ == "__main__":
    main()
