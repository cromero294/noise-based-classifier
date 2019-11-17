# sys.path.append('/home/cromero/noise-based-classifier/')

import datasets.DatasetGenerator as data
from resources.PlotModel import *

from src.ClasificadorRuido import *
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm


def main():

    model = "threenorm"

    a, b, y_test = data.create_dataset(100, model)

    X_test = np.array(np.c_[a, b])
    y_test = np.array(y_test)[:, 0]


    #########################################
    #####      DATA CLASSIFICATION      #####
    #########################################

    n_trees = 100

    for perc in tqdm(np.arange(0.1, 0.91, 0.1)):

        ### Classifier generation ###

        clf = ClasificadorRuido(n_trees=n_trees, perc=perc)

        clf_scores = np.zeros((100, n_trees))

        for i in range(100):
            ### Training data generation ###

            a, b, y_train = data.create_dataset(300, model)

            X_train = np.array(np.c_[a, b])
            y_train = np.array(y_train)[:, 0]

            ### Classifiers training and classification ###

            clf.fit(X_train, y_train, random_perc=False)
            clf.predict_proba_error(X_test)

            for n_tree in range(1, n_trees + 1):
                clf_scores[i, n_tree - 1] += 1 - clf.score_error(X_test, y_test, n_classifiers=n_tree)

        plt.plot(range(1, n_trees + 1), clf_scores.mean(axis=0), linestyle='-.')
        print()
        print("Perc: ", np.round(perc, 1), " - ", clf_scores.mean(axis=0)[-1])

    ### Random Forest ###

    rfclf = RandomForestClassifier(n_estimators=n_trees)

    a, b, y_train = data.create_dataset(300, model)

    X_train = np.array(np.c_[a, b])
    y_train = np.array(y_train)[:, 0]

    rfclf.fit(X_train, y_train)
    rfscore = 1 - rfclf.score(X_test, y_test)

    plt.axhline(y=rfscore, color='m', linestyle='-')

    plt.legend(('perc=0.1', 'perc=0.2', 'perc=0.3', 'perc=0.4', 'perc=0.5', 'perc=0.6', 'perc=0.7', 'perc=0.8', 'perc=0.9', 'RF'),
               loc='upper right')

    plt.savefig("../plots/noise-error_"+model+"_100trees.png")


if __name__ == "__main__":
    main()
