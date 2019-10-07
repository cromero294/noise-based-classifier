import sys
import lib.properties as properties

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import numpy as np
import pandas as pd
import pandas_ml as pml
import matplotlib.pyplot as plt

from src.ClasificadorRuido import *


def get_data():
    if len(sys.argv) > 1:
        try:
            dataset = pd.read_csv(properties.DATASET_DIRECTORY + sys.argv[1])
            dataset = dataset.values
            X, y = dataset[:,:-1], dataset[:,-1]
            # TODO: X and y must be non categorical values
        except IOError:
            print "File \"{}\" does not exist.".format(sys.argv[1])
            return
    else:
        X, y = make_moons(n_samples=10000, shuffle=True, noise=0.5, random_state=None)

    return X, y


def main():
    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = ClasificadorRuido(n_trees=100, perc=0.5)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test, suggested_class=None)

    confusion_matrix = pml.ConfusionMatrix(y_test, y_pred)
    print "Confusion matrix:\n%s" % confusion_matrix

    confusion_matrix.plot()
    plt.show()


if __name__ == "__main__":
    main()
