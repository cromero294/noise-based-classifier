import sys

sys.path.append('/home/cromero/noise-based-classifier/')

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import resources.properties as properties
import datasets.DatasetGenerator as data
import pandas as pd

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from src.ClasificadorRuido import *
from src.Alfredo import *

from tqdm import tqdm


model = 'ringnorm'

X, y = data.create_full_dataset(5000, 21, model)

# y = y.ravel()

print(y)

X = np.append(X, y, axis=1)

print(X)

np.savetxt(properties.DATASET_DIRECTORY + "csv/ringnorm2.csv", X, delimiter=",", fmt='%1.5f')
