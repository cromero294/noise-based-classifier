import sys

sys.path.append('/home/cromero/noise-based-classifier/')

import datasets.DatasetGenerator as data
import resources.properties as properties

from src.Alfredo import *
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm


def main():

    model = "threenorm"

    X_test, y_test = data.create_full_dataset(5000, 20, model)

    y_test = y_test.transpose()[0]


    #########################################
    #####      DATA CLASSIFICATION      #####
    #########################################

    n_trees = 100
    times = 100

    rf_scores = []
    clf_scores = np.empty((times, len(np.arange(0.01, 0.99, 0.01)), n_trees))

    for i in tqdm(range(times)):
        ### Training data generation ###

        X_train, y_train = data.create_full_dataset(300, 20, model)

        y_train = y_train.transpose()[0]

        ### Classifiers training and classification ###

        # RANDOM FOREST

        rfclf = RandomForestClassifier(n_estimators=n_trees)

        rfclf.fit(X_train, y_train)
        rf_scores.append(1 - rfclf.score(X_test, y_test))

        # NOISE BASED

        for perci, perc in enumerate(np.arange(0.01, 0.99, 0.01)):
            clf = Alfredo(n_trees=n_trees, perc=perc, bagg=True)

            clf.fit(X_train, y_train, random_perc=False)
            clf.predict_proba_error(X_test)

            scores = []

            for n_tree in range(1, n_trees + 1):
                scores.append(1 - clf.score_error(X_test, y_test, n_classifiers=n_tree))

            clf_scores[i, perci] = np.array(scores)

    print(np.array(rf_scores))
    print()
    print(clf_scores)
    print(clf_scores.shape)
    print()
    print(clf_scores.mean(axis=0))
    print(clf_scores.mean(axis=0).shape)

    np.save(properties.DATA + properties.SYNTHETIC + model + "_data_random-forest_ALFREDO", np.array(rf_scores))
    np.save(properties.DATA + properties.SYNTHETIC + model + "_data_ALFREDO", clf_scores)


if __name__ == "__main__":
    main()
