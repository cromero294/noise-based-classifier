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

from src.ClasificadorRuidoBagging import *


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
        X, y = make_moons(n_samples=1000, shuffle=True, noise=0.5, random_state=42)

    return X, y


def main():
    X, y = get_data()

    # X_tr, X_te, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_tr, X_te, y_train, y_test = train_test_split(X, y, test_size=0.2)

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

    # X_train = pca.fit_transform(X_tr_std)
    # X_test = pca.transform(X_te_std)

    X_train = X_tr
    X_test = X_te

    #########################################
    #####      DATA CLASSIFICATION      #####
    #########################################
    n_trees = 1000

    clf_bagg = ClasificadorRuidoBagging(n_trees=n_trees, perc=0.4, sub_train=0.3)
    clf = ClasificadorRuido(n_trees=n_trees, perc=0.5)
    rfclf = RandomForestClassifier(n_estimators=n_trees)
    boostingclf = GradientBoostingClassifier(n_estimators=n_trees)
    baggingclf = BaggingClassifier(n_estimators=n_trees)

    clf_bagg.fit(X_train, y_train, random_perc=False)
    clf_bagg.predict(X_test, suggested_class=None)

    clf.fit(X_train, y_train, random_perc=False)
    clf.predict(X_test, suggested_class=None)

    rfclf.fit(X_train, y_train)
    rfclf.predict(X_test)

    boostingclf.fit(X_train, y_train)
    boostingclf.predict(X_test)

    baggingclf.fit(X_train, y_train)
    baggingclf.predict(X_test)

    print("----------------------------------------------")
    print("{} Score + BAGGING:{} {}".format(properties.COLOR_BLUE, properties.END_C, clf_bagg.score(X_test, y_test, suggested_class=None)))
    print("{} Score:{} {}".format(properties.COLOR_BLUE, properties.END_C, clf.score(X_test, y_test, suggested_class=None)))
    print("{} Random forest score:{} {}".format(properties.COLOR_BLUE, properties.END_C, rfclf.score(X_test, y_test)))
    print("{} Boosting score:{} {}".format(properties.COLOR_BLUE, properties.END_C, boostingclf.score(X_test, y_test)))
    print("{} Bagging score:{} {}".format(properties.COLOR_BLUE, properties.END_C, baggingclf.score(X_test, y_test)))

    plt.subplot(1, 2, 1)
    plot_model(clf, X_train, y_train, "Noise based")

    plt.plot()

    plt.subplot(1, 2, 2)
    plot_model(clf_bagg, X_train, y_train, "Noise BAGGING")

    plt.plot()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
