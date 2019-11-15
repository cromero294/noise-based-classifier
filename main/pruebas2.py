# sys.path.append('/home/cromero/noise-based-classifier/')

import datasets.DatasetGenerator as data
from resources.PlotModel import *

from src.ClasificadorRuido import *
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm


def main():

    model = "ringnorm"

    a, b, y_test = data.create_dataset(5000, model)

    X_test = np.array(np.c_[a, b])
    y_test = np.array(y_test)[:, 0]


    #########################################
    #####      DATA CLASSIFICATION      #####
    #########################################

    n_trees = 100
    times = 100

    scores_clf = []

    for perc in tqdm(np.arange(0.1, 0.9, 0.01)):

        ### Classifier generation ###

        clf = ClasificadorRuido(n_trees=n_trees, perc=perc)

        score_clf = 0

        for _ in range(times):
            ### Training data generation ###

            a, b, y_train = data.create_dataset(300, model)

            X_train = np.array(np.c_[a, b])
            y_train = np.array(y_train)[:, 0]

            ### Classifiers training and classification ###

            clf.fit(X_train, y_train, random_perc=False)

            score_clf += (1 - clf.score(X_test, y_test))

        scores_clf.append(score_clf / times)

    ### Random Forest ###

    rfclf = RandomForestClassifier(n_estimators=n_trees)

    rfscore = 0

    for _ in range(times):
        a, b, y_train = data.create_dataset(300, model)

        X_train = np.array(np.c_[a, b])
        y_train = np.array(y_train)[:, 0]

        rfclf.fit(X_train, y_train)
        rfscore += 1 - rfclf.score(X_test, y_test)

    plt.axhline(y=rfscore/times, color='m', linestyle='-')

    plt.plot(np.arange(0.1, 0.9, 0.01), scores_clf, linestyle='-.')

    plt.savefig("../plots/noise-variation_ringnorm.png")


if __name__ == "__main__":
    main()
