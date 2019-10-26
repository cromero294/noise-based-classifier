# -*- coding: utf-8 -*-

from __future__ import division
from resources.plotModel import *
from resources.Datos import *
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

try:
    dataset = Datos("Datasets/tic-tac-toe.data")

    X = dataset.getDatos()[:,:-1] # Todos los atributos menos la clase
    y = dataset.getDatos()[:,-1] # Todas las clases

    k_folds = 10

    skf = StratifiedKFold(n_splits=k_folds)

    score_tree = []
    score_0 = []
    score_1 = []
    score_both = []
    score_random = []

    clfTree = tree.DecisionTreeClassifier()
    clf = ClasificadorRuido()
    clfRandom = RandomForestClassifier(n_estimators=101)

    for train_index, test_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        Xt, yt = X[test_index], y[test_index]

        clfTree = clfTree.fit(X_train, y_train)
        clfRandom.fit(X_train,y_train)
        clf.fit(X_train, y_train)

        score_tree.append(1 - clfTree.score(Xt, yt))
        score_0.append(1 - clf.score(Xt, yt, 0))
        score_1.append(1 - clf.score(Xt, yt, 1))
        score_both.append(1 - clf.score(Xt, yt))
        score_random.append(1 - clfRandom.score(Xt, yt))

    print "Tree:\n  Err:\t" + str(round(np.mean(score_tree),4)) + " ±" + str(round(np.std(score_tree),4))
    print "  Tasa de acierto:\t" + str(round(1 - np.mean(score_tree),4))
    print "Ceros:\n  Err:\t" + str(round(np.mean(score_0),4)) + " ±" + str(round(np.std(score_0),4))
    print "  Tasa de acierto:\t" + str(round(1 - np.mean(score_0),4))
    print "Unos:\n  Err:\t" + str(round(np.mean(score_1),4)) + " ±" + str(round(np.std(score_1),4))
    print "  Tasa de acierto:\t" + str(round(1 - np.mean(score_1),4))
    print "0 - 1:\n  Err:\t" + str(round(np.mean(score_both),4)) + " ±" + str(round(np.std(score_both),4))
    print "  Tasa de acierto:\t" + str(round(1 - np.mean(score_both),4))
    print "Random forest:\n  Err:\t" + str(round(np.mean(score_random),4)) + " ±" + str(round(np.std(score_random),4))
    print "  Tasa de acierto:\t" + str(round(1 - np.mean(score_random),4))

except ValueError as e:
    print(e)
