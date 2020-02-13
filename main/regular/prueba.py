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


def get_data():
    try:
        dataset = pd.read_csv(properties.DATASET_DIRECTORY + sys.argv[1])
        cat_columns = dataset.select_dtypes(['object']).columns

        # AUXILIAR


        # aux = dataset['class']
        # dataset.drop(labels=['class'], axis=1, inplace = True)
        # dataset.insert(13, 'class', aux)
        #
        cat_columns = ['class']
        # print(dataset)

        # AUXILIAR

        dataset[cat_columns] = dataset[cat_columns].astype('category')
        dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)
        dataset = dataset.values
        X, y = dataset[:, :-1], dataset[:, -1]

        return X, y
    except IOError:
        print("File \"{}\" does not exist.".format(sys.argv[1]))
        return

model = 'ringnorm'

X, y = data.create_full_dataset(5000, 21, model)

y = y.ravel()

# model = sys.argv[1]
# X, y = get_data()

n_trees = 100
k_folds = 100

skf = StratifiedShuffleSplit(n_splits=k_folds, test_size=0.33, random_state=32)

# np.save(properties.DATA + properties.DATASETS + model + "_data_random-forest", np.array(rf_scores))

train = []
test = []

for i, (train_index, test_index) in tqdm(enumerate(skf.split(X, y))):
    train.append(train_index)
    test.append(test_index)

np.save(properties.DATA + properties.STRATIFIED + model + "_train", np.array(train))
np.save(properties.DATA + properties.STRATIFIED + model + "_test", np.array(test))

train_prueba = np.load(properties.DATA + properties.STRATIFIED + model + "_train.npy")
test_prueba = np.load(properties.DATA + properties.STRATIFIED + model + "_test.npy")

print(train_prueba)
print(train_prueba.shape)
print(test_prueba)
print(test_prueba.shape)
