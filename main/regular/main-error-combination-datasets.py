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

def get_data(model):
        try:
            dataset = pd.read_csv(properties.DATASET_DIRECTORY + "csv/" + model + ".csv")
            # print(dataset.dtypes)

            # AUXILIAR


            # aux = dataset['class']
            # dataset.drop(labels=['class'], axis=1, inplace = True)
            # dataset.insert(5, 'class', aux)

            cat_columns = ['class']

            if model == "tic-tac-toe":
                cat_columns = dataset.select_dtypes(['object']).columns

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

def main():

    model = sys.argv[1]

    X, y = get_data(model)

    # model = 'twonorm'
    #
    # X, y = data.create_full_dataset(5300, 20, model)
    #
    # y = y.ravel()

    #########################################
    #####      DATA CLASSIFICATION      #####
    #########################################

    n_trees = 100
    k_folds = 100

    # skf = StratifiedShuffleSplit(n_splits=k_folds, test_size=0.8)

    train = np.load(properties.DATA + properties.STRATIFIED + model + "_train.npy")
    test = np.load(properties.DATA + properties.STRATIFIED + model + "_test.npy")

    rf_scores = []
    tree_scores = []
    bagg_scores = []
    boost_scores = []

    clf_scores = np.empty((k_folds, len(np.arange(0.01, 0.99, 0.01)), n_trees))

    print(model)

    for i in tqdm(range(k_folds)):
        train_index = train[i, :]
        test_index = test[i, :]

        ### Training data generation ###

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # print(X_train, y_train)

        ### Classifiers training and classification ###

        # RANDOM FOREST

        rfclf = RandomForestClassifier(n_estimators=n_trees)
        tree_clf = tree.DecisionTreeClassifier()
        boosting = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=5), n_estimators=n_trees)
        bagging = BaggingClassifier(n_estimators=n_trees)

        rfclf.fit(X_train, y_train)
        rf_scores.append(1 - rfclf.score(X_test, y_test))

        tree_clf.fit(X_train, y_train)
        tree_scores.append(1 - tree_clf.score(X_test, y_test))

        boosting.fit(X_train, y_train)
        boost_scores.append(1 - boosting.score(X_test, y_test))

        bagging.fit(X_train, y_train)
        bagg_scores.append(1 - bagging.score(X_test, y_test))

        # NOISE BASED

        for perci, perc in enumerate(np.arange(0.01, 0.99, 0.01)):
            clf = Alfredo(n_trees=n_trees, perc=perc, bagg=True)

            clf.fit(X_train, y_train, random_perc=False)
            clf.predict_proba_error(X_test)

            scores = []

            for n_tree in range(1, n_trees + 1):
                scores.append(1 - clf.score_error(X_test, y_test, n_classifiers=n_tree))

            clf_scores[i, perci] = np.array(scores)


    np.save(properties.DATA + properties.DATASETS + model + "_data_random-forest", np.array(rf_scores))
    np.save(properties.DATA + properties.DATASETS + model + "_data_tree", np.array(tree_scores))
    np.save(properties.DATA + properties.DATASETS + model + "_data_boosting", np.array(boost_scores))
    np.save(properties.DATA + properties.DATASETS + model + "_data_bagging", np.array(bagg_scores))
    np.save(properties.DATA + properties.DATASETS + model + "_data", clf_scores)


if __name__ == "__main__":
    main()
