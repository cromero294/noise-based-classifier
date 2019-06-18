#-*- coding: utf-8 -*-
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
        self.clasificadores = []

        for epoca in range(self.nepocas):
            clfTree = tree.DecisionTreeClassifier()
            X_cambiado, y_cambiado = self.change_class(x, y)
            clfTree.fit(X_cambiado, y_cambiado)
            self.clasificadores.append(clfTree)

    def score(self, x, y, class_atrib=None):
        aciertos = 0

        pre = self.predict(x, class_atrib)

        for i,pred in enumerate(pre):
            if pred == y[i]:
                aciertos += 1.

        return aciertos/x.shape[0]

    def predict(self, x, class_atrib=None):
        prediccs = []

        for pred in self.predict_proba(x, class_atrib):
            if pred[0] > pred[1]:
                prediccs.append(0.)
            else:
                prediccs.append(1.)

        return prediccs

    def predict_proba(self, x, class_atrib=None):
        clasificacion = []
        clasificacion_final = [[0, 0] for i in range(x.shape[0])]

        if class_atrib == None:
            probs1 = self.predict_proba(x, 1)
            probs0 = self.predict_proba(x, 0)

            for i in range(x.shape[0]):
                clasificacion_final[i][0] += (probs0[i][0] + probs1[i][0])/2
                clasificacion_final[i][1] += (probs0[i][1] + probs1[i][1])/2

        else:
            if class_atrib == 1:
                datos = np.ones((x.shape[0], x.shape[1]+1))
            elif class_atrib == 0:
                datos = np.zeros((x.shape[0], x.shape[1]+1))

            datos[:,:-1] = x
            x = datos

            pred = []

            for clasificador in self.clasificadores:
                aux = clasificador.predict_proba(x)
                for i,clf in enumerate(aux):
                    if class_atrib == 1:
                        clasificacion_final[i][1] += clf[1]
                        clasificacion_final[i][0] += clf[0]
                    elif class_atrib == 0:
                        clasificacion_final[i][1] += clf[0]
                        clasificacion_final[i][0] += clf[1]

            for i in range(x.shape[0]):
                clasificacion_final[i][0] /= len(self.clasificadores)
                clasificacion_final[i][1] /= len(self.clasificadores)

        return clasificacion_final

    def predict_proba_error(self, x, class_atrib=None):
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

            for clasificador in self.clasificadores:
                aux = clasificador.predict_proba(x)
                if class_atrib == 1:
                    self.pred_unos.append(aux)
                elif class_atrib == 0:
                    self.pred_ceros.append(aux)

    def score_error(self, x, y, n_classifiers=None, class_atrib=None):
        if n_classifiers == None:
            n_classifiers = len(self.clasificadores)

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

        datos = np.c_[x, y]

        numDatos = datos.shape[0]
        porcentaje = int(numDatos * self.perc)

        datos_nuevos = datos.copy()

        arrayAleatorio = range(0, numDatos)

        shuffle(arrayAleatorio)

        for num in arrayAleatorio[:porcentaje]:
            datos_nuevos[num,-1] = 1 - datos_nuevos[num,-1]

        clase_bien_mal_clasificado = [(1.0 if datos_nuevos[i,-1] == datos[i,-1] else 0.0) for i in range(0,numDatos)]

        return datos_nuevos, np.array(clase_bien_mal_clasificado)
