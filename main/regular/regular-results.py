import sys

sys.path.append('/home/cromero/noise-based-classifier/')

import resources.properties as properties
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from src.ClasificadorRuido import *
from src.Alfredo import *

def main():
    model = "heart.dat"

    data_alfredo = np.load(properties.DATA + properties.SCORES + model + "_ALFREDO-SCORES.npy")
    data_bagging = np.load(properties.DATA + properties.SCORES + model + "_BAGGING-SCORES.npy")
    data_boosting = np.load(properties.DATA + properties.SCORES + model + "_BOOSTING-SCORES.npy")
    data_noise = np.load(properties.DATA + properties.SCORES + model + "_NOISE-SCORES.npy")
    data_rf = np.load(properties.DATA + properties.SCORES + model + "_RF-SCORES.npy")
    data_tree = np.load(properties.DATA + properties.SCORES + model + "_TREE-SCORES.npy")

    print("MODELO: " + model)
    print()
    print("\t" + properties.COLOR_BLUE + "ALFREDO: " + properties.END_C + str(data_alfredo.mean()))
    print("\t" + properties.COLOR_BLUE + "BAGGING: " + properties.END_C + str(data_bagging.mean()))
    print("\t" + properties.COLOR_BLUE + "BOOSTING: " + properties.END_C + str(data_boosting.mean()))
    print("\t" + properties.COLOR_BLUE + "NOISE: " + properties.END_C + str(data_noise.mean()))
    print("\t" + properties.COLOR_BLUE + "RF: " + properties.END_C + str(data_rf.mean()))
    print("\t" + properties.COLOR_BLUE + "TREE: " + properties.END_C + str(data_tree.mean()))


if __name__ == "__main__":
    main()
