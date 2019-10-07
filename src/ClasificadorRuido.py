# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from random import *
from sklearn import tree


class ClasificadorRuido:

    def __init__(self, n_trees=100, perc=0.5):
        self.n_trees = n_trees
        self.perc = perc

    def fit(self, x, y):
        """
        This method is used to fit each one of the decision trees the random noise classifier is composed with.
        This is the way to fit the complete classifier and it is compulsory to carry on with the data classification.

        :param x: original features from the dataset
        :param y: original classes from the dataset
        """
        self.classifiers = []

        for classifier in range(self.n_trees):
            tree_clf = tree.DecisionTreeClassifier()
            modified_x, modified_y = self.change_class(x, y)
            tree_clf.fit(modified_x, modified_y)
            self.classifiers.append(tree_clf)

    def score(self, x, y, suggested_class=None):
        """
        This method is used to calculate the classifier accuracy comparing the obtained classes with the original
        ones from the dataset.

        :param x: original features from the dataset
        :param y: original classes from the dataset
        :param suggested_class: a new feature added to classify the examples
        :return: classifier accuracy
        """
        return sum([1 for i, prediction in enumerate(self.predict(x, suggested_class)) if prediction == y[i]])/x.shape[0]

    def predict(self, x, suggested_class=None):
        """
        This method is used to generate the class predictions from each example to be classified.
        It uses the method predict_proba to calculate the probabilities that a data is well classified or not.

        :param x: original features from the dataset
        :param suggested_class: a new feature added to classify the examples
        :return: an array with the predicted class for each example from the dataset
        """
        return np.array([0. if pred[0] > pred[1] else 1. for pred in self.predict_proba(x, suggested_class)])

    def predict_proba(self, x, suggested_class=None):
        """
        This method calculates the probability that a data is well classified or not. It adds a new feature
        to the dataset depending on the suggested_class attribute.

        :param x: data to be classified
        :param suggested_class: new feature to be added
        :return: probabilities that a data is well classified or not
        """
        predictions = []

        if suggested_class is None:
            probs1 = self.predict_proba(x, 1)
            probs0 = self.predict_proba(x, 0)

            predictions = (probs0 + probs1) / 2

        else:
            if suggested_class == 1:
                data = np.ones((x.shape[0], x.shape[1]+1))
            elif suggested_class == 0:
                data = np.zeros((x.shape[0], x.shape[1]+1))

            data[:, :-1] = x
            x = data

            # It creates a numpy array out of the mean of the classifications obtained from each single classifier
            [predictions.append(clf.predict_proba(x)) for clf in self.classifiers]
            predictions = np.array(predictions).mean(axis=0)

            # If the suggested_class param is '0', it means the "well classified" class must be exchanged with the
            # "poorly classified" class
            if suggested_class == 0:
                predictions[:, [0, 1]] = predictions[:, [1, 0]]

        return predictions

    def predict_proba_error(self, x, suggested_class=None):
        """
        This method calculates a matrix which contains the probabilities of each example cumulatively.

        :param x: the original features from the dataset
        :param suggested_class: the class the classifier uses as new feature
        :return: the final probabilities matrix
        """
        if suggested_class is None:
            probs1 = self.predict_proba_error(x, suggested_class=1)
            probs0 = self.predict_proba_error(x, suggested_class=0)

            self.predictions = (probs0 + probs1) / 2

        else:
            if suggested_class == 1:
                data = np.ones((x.shape[0], x.shape[1]+1))
            elif suggested_class == 0:
                data = np.zeros((x.shape[0], x.shape[1]+1))

            data[:, :-1] = x
            x = data

            self.predictions = []

            [self.predictions.append(clf.predict_proba(x)) for clf in self.classifiers]
            self.predictions = np.array(self.predictions)

            for i in range(len(self.classifiers)-1, -1, -1):
                self.predictions[i, :, :] = (self.predictions[:i+1, :, :].sum(axis=0))
                self.predictions[i, :, :] /= i+1

            if suggested_class == 0:
                self.predictions[:, :, [0, 1]] = self.predictions[:, :, [1, 0]]

        return self.predictions

    def score_error(self, x, y, n_classifiers=100):
        """
        With this method we are able to see what is going on with the classification of the examples for each classifier.
        This method allows us to calculate the score obtained using the amount of classifiers we want up to the maximum
        of classifiers with which it was declared.

        :param x: original features dataset
        :param y: original classes from the dataset
        :param n_classifiers: number of classifiers used to calculate the score
        :return: score obtained
        """
        if n_classifiers is None:
            n_classifiers = len(self.classifiers)

        n_classifiers -= 1

        return sum([1 for i, pred in enumerate(self.predictions[n_classifiers, :, :]) if (pred[0] > pred[1] and y[i] == 0) or (pred[1] >= pred[0] and y[i] == 1)]) / x.shape[0]

    def change_class(self, x, y):
        """
        Given a data set split in features and classes this method transforms this set into another set.
        This new set is created based on random noise generation and its classification. The randomization
        of the new data set is given by the percentage received from the class constructor.

        The randomization is generated changing some data classes and pointing it out in the new class.
        The new class is calculated comparing the original class with the data randomization. For this new class
        '1' means "well classified" and '0', the opposite.

        :param x: features from the original data set
        :param y: classes from the original data set
        :return: features and classes from the new data set
        """
        data = np.c_[x, y]

        num_data = data.shape[0]
        percentage = int(num_data * self.perc)

        updated_data = data.copy()

        random_data = range(0, num_data)
        shuffle(random_data)

        for num in random_data[:percentage]:
            updated_data[num, -1] = 1 - updated_data[num, -1]

        updated_class = [(updated_data[i, -1] == data[i, -1]) for i in range(0, num_data)]

        return updated_data, np.array(updated_class)