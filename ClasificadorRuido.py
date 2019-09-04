#-*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import re
import sys
from random import *
from sklearn import tree
from scipy import stats

class ClasificadorRuido:

    def __init__(self, n_trees=101, perc=0.5):
        self.nepocas = n_trees
        self.perc = perc

    def fit(self, x, y):
        self.classifiers = []

        for epoca in range(self.nepocas):
            clfTree = tree.DecisionTreeClassifier()
            X_cambiado, y_cambiado = self.change_class(x, y)
            clfTree.fit(X_cambiado, y_cambiado)
            self.classifiers.append(clfTree)

    def score(self, x, y, suggested_class=None):
        return sum([1 for i,prediction in enumerate(self.predict(x, suggested_class)) if prediction == y[i]])/x.shape[0]

    def predict(self, x, suggested_class=None):
        predictions = []

        for pred in self.predict_proba(x, suggested_class):
            if pred[0] > pred[1]:
                predictions.append(0.)
            else:
                predictions.append(1.)

        return np.array(predictions)

    def predict_proba(self, x, suggested_class=None):
        """
        This method calculates the probability that a data is well classified or not. It adds a new feature
        to the dataset depending on the suggested_class attribute.

        :param x: data to be classified
        :param suggested_class: new attribute to be added
        :return: probabilities that a data is well classified or not
        """
        predictions = []

        if suggested_class == None:
            probs1 = self.predict_proba(x, 1)
            probs0 = self.predict_proba(x, 0)

            predictions = (probs0 + probs1) / 2

        else:
            if suggested_class == 1:
                data = np.ones((x.shape[0], x.shape[1]+1))
            elif suggested_class == 0:
                data = np.zeros((x.shape[0], x.shape[1]+1))

            data[:,:-1] = x
            x = data

            # It creates a numpy array out of the mean of the classifications obtained from each single classifier
            [predictions.append(clf.predict_proba(x)) for clf in self.classifiers]
            predictions = np.array(predictions).mean(axis=0)

            # If the suggested_class param is '0', it means the "well classified" class must be exchanged with the
            # "poorly classified" class
            if suggested_class == 0:
                predictions[:, [0, 1]] = predictions[:, [1, 0]]

        return predictions

    def predict_proba_error(self, x, class_atrib=None):
        # TODO: create an accumulative array out of the classifications just adding them instead of classifying it time after time
        clasificacion = []

        if class_atrib == None:
            probs1 = self.predict_proba_error(x, 1)
            probs0 = self.predict_proba_error(x, 0)

            self.pred = []
            for z,y in zip(self.pred_ceros,self.pred_unos):
                predaux = []
                for i in range(len(z)):
                    predaux.append([(z[i][1] + y[i][0])/2, (z[i][0] + y[i][1])/2])
                self.pred.append(predaux)

        else:
            if class_atrib == 1:
                datos = np.ones((x.shape[0], x.shape[1]+1))
                self.pred_unos = []
            elif class_atrib == 0:
                datos = np.zeros((x.shape[0], x.shape[1]+1))
                self.pred_ceros = []

            datos[:,:-1] = x
            x = datos

            for clasificador in self.classifiers:
                aux = clasificador.predict_proba(x)
                if class_atrib == 1:
                    self.pred_unos.append(aux)
                elif class_atrib == 0:
                    self.pred_ceros.append(aux)

    def score_error(self, x, y, n_classifiers=None, class_atrib=None):
        if n_classifiers == None:
            n_classifiers = len(self.classifiers)

        aux = []

        if class_atrib == 1:
            predaux = self.pred_unos
        if class_atrib == 0:
            predaux = self.pred_ceros
        if class_atrib == None:
            predaux = self.pred

        clasificacion_final = [[0, 0] for i in range(x.shape[0])]

        for pred in predaux[:n_classifiers]:
            for i,dato in enumerate(pred):
                if class_atrib != 0:
                    clasificacion_final[i][1] += dato[1]
                    clasificacion_final[i][0] += dato[0]
                else:
                    clasificacion_final[i][1] += dato[0]
                    clasificacion_final[i][0] += dato[1]

        aciertos = 0

        for i in range(x.shape[0]):
            clasificacion_final[i][0] /= n_classifiers
            clasificacion_final[i][1] /= n_classifiers

            prediccion_auxiliar = 1
            if clasificacion_final[i][0] > clasificacion_final[i][1]:
                prediccion_auxiliar = 0

            if prediccion_auxiliar == y[i]:
                aciertos += 1.0

        return aciertos/x.shape[0]

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
