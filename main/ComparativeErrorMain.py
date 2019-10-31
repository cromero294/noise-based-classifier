import sys

sys.path.append('/home/cromero/noise-based-classifier/')

import resources.properties as properties
from resources.PlotModel import *

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

import pandas as pd

from src.ClasificadorRuido import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm


def get_data():
    if len(sys.argv) > 1:
        try:
            dataset = pd.read_csv(properties.DATASET_DIRECTORY + sys.argv[1])
            cat_columns = dataset.select_dtypes(['object']).columns
            dataset[cat_columns] = dataset[cat_columns].astype('category')
            dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)
            dataset = dataset.values
            X, y = dataset[:,:-1], dataset[:,-1]
        except IOError:
            print("File \"{}\" does not exist.".format(sys.argv[1]))
            return
    else:
        X, y = make_moons(n_samples=10000, shuffle=True, noise=0.5, random_state=None)

    return X, y


def main():
    X, y = get_data()

    X_tr, X_te, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

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

    #########################################
    #####      DATA CLASSIFICATION      #####
    #########################################
    rfclf = RandomForestClassifier(n_estimators=1000)

    rfclf.fit(X_train, y_train)
    rfscore = 1 - rfclf.score(X_test, y_test)

    plt.axhline(y=rfscore, color='m', linestyle='--')

    n_trees = 100

    colors = ['c', 'k', 'y', 'b']

    for x_perc, perc in enumerate([0.2, 0.5, 0.7, 0.9]):
        clf_scores = np.zeros((n_trees, n_trees))

        clf = ClasificadorRuido(n_trees=n_trees, perc=perc)

        for i in tqdm(range(100)):

            clf.fit(X_train, y_train)
            clf.predict_proba_error(X_test)

            for n_tree in range(1, n_trees+1):
                clf_scores[i, n_tree-1] += 1 - clf.score_error(X_test, y_test, n_classifiers=n_tree)

        plt.plot(range(1, n_trees+1), clf_scores.mean(axis=0), linestyle='-.', color=colors[x_perc])

    plt.legend(('RF', 'perc=0.25', 'perc=0.5', 'perc=0.75', 'perc=0.9'), loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()
