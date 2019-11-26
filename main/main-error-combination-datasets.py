import sys

sys.path.append('/home/cromero/noise-based-classifier/')

import datasets.DatasetGenerator as data
from resources.PlotModel import *
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import resources.properties as properties
import pandas as pd

from src.ClasificadorRuido import *
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm


def main():
    try:
        dataset = pd.read_csv(properties.DATASET_DIRECTORY + sys.argv[1])
        cat_columns = dataset.select_dtypes(['object']).columns
        dataset[cat_columns] = dataset[cat_columns].astype('category')
        dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)
        dataset = dataset.values
        X, y = dataset[:, :-1], dataset[:, -1]
    except IOError:
        print("File \"{}\" does not exist.".format(sys.argv[1]))
        return

    #########################################
    #####      DATA CLASSIFICATION      #####
    #########################################

    n_trees = 1000
    k_folds = 10

    skf = StratifiedKFold(n_splits=k_folds)

    rf_scores = []
    clf_scores = np.empty((k_folds, len(np.arange(0.01, 0.99, 0.01)), n_trees))

    for i, (train_index, test_index) in tqdm(enumerate(skf.split(X, y))):

        ### Training data generation ###

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        ### Classifiers training and classification ###

        # RANDOM FOREST

        rfclf = RandomForestClassifier(n_estimators=n_trees)

        rfclf.fit(X_train, y_train)
        rf_scores.append(1 - rfclf.score(X_test, y_test))

        # NOISE BASED

        for perci, perc in enumerate(np.arange(0.01, 0.99, 0.01)):
            clf = ClasificadorRuido(n_trees=n_trees, perc=perc)

            clf.fit(X_train, y_train, random_perc=False)
            clf.predict_proba_error(X_test)

            scores = []

            for n_tree in range(1, n_trees + 1):
                scores.append(1 - clf.score_error(X_test, y_test, n_classifiers=n_tree))

            clf_scores[i, perci] = np.array(scores)

    np.save("../data/" + sys.argv[1] + "_data_random-forest", np.array(rf_scores))
    np.save("../data/" + sys.argv[1] + "_data", clf_scores)


if __name__ == "__main__":
    main()
