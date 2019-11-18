# sys.path.append('/home/cromero/noise-based-classifier/')

import datasets.DatasetGenerator as data
from resources.PlotModel import *

from src.ClasificadorRuido import *
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm


def main():

    model = "ringnorm"

    a, b, y_test = data.create_dataset(100, model)

    X_test = np.array(np.c_[a, b])
    y_test = np.array(y_test)[:, 0]


    #########################################
    #####      DATA CLASSIFICATION      #####
    #########################################

    n_trees = 100

    clf_scores = np.empty((100, len(np.arange(0.01, 0.99, 0.01)), n_trees))

    for i in tqdm(range(1000)):
        ### Training data generation ###

        a, b, y_train = data.create_dataset(300, model)

        X_train = np.array(np.c_[a, b])
        y_train = np.array(y_train)[:, 0]

        ### Classifiers training and classification ###

        for perci, perc in enumerate(np.arange(0.01, 0.99, 0.01)):
            clf = ClasificadorRuido(n_trees=n_trees, perc=perc)

            clf.fit(X_train, y_train, random_perc=False)
            clf.predict_proba_error(X_test)

            scores = []

            for n_tree in range(1, n_trees + 1):
                scores.append(1 - clf.score_error(X_test, y_test, n_classifiers=n_tree))

            clf_scores[i, perci] = np.array(scores)

    print(clf_scores)
    print(clf_scores.shape)
    print()
    print(clf_scores.mean(axis=1))
    print(clf_scores.mean(axis=1).shape)

    np.save("../data/"+ model +"_data", clf_scores)


if __name__ == "__main__":
    main()
