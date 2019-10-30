import sys

sys.path.append('/home/cromero/noise-based-classifier/')

import resources.properties as properties
from resources.PlotModel import *

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

import numpy as np
import pandas as pd
import pandas_ml as pml
import matplotlib.pyplot as plt

from src.ClasificadorRuido import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = ClasificadorRuido(n_trees=100, perc=0.5)
    rfclf = RandomForestClassifier(n_estimators=100)

    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X_test)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test, suggested_class=None)

    rfclf.fit(X_train, y_train)
    rf_y_pred = rfclf.predict(X_test)

    confusion_matrix = pml.ConfusionMatrix(y_test, y_pred)
    print("----------------------------------------------")
    print("{} Confusion matrix:\n{}\n{}\n".format(properties.COLOR_BLUE, properties.END_C, confusion_matrix))
    print("{} Score:{} {}".format(properties.COLOR_BLUE, properties.END_C, clf.score(X_test, y_test, suggested_class=None)))
    print("{} Random forest score:{} {}".format(properties.COLOR_BLUE, properties.END_C, rfclf.score(X_test, y_test)))

    confusion_matrix.plot()
    plt.show()

    plot_model(clf, X_test, y_pred, y_test)


if __name__ == "__main__":
    main()
