# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:13:00 2018

@author: Mario Calle Romero
"""
from __future__ import division
from lib.Datos import Datos
from src.ClasificadorRuido import ClasificadorRuido
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

try:
    k_folds = 10
    num_trees = 100

    skf = StratifiedKFold(n_splits=k_folds)

    clf = ClasificadorRuido()
    clfRandom = RandomForestClassifier(n_estimators=num_trees)

    clase_atrib = [0, 1, None]
    linestyle = ['-.', ':', '-']
    color = ['red', 'olive', 'skyblue']
    datasets = ["tic-tac-toe.data","german.data","wdbc.data","magic04.data","diabetes.csv","online_shoppers_intention.csv", "bank-full.csv", "EEG_Eye_State.arff"]

    fig, axs = plt.subplots(4, 2, figsize=(10, 12), tight_layout=True)

    n = 1
    m = -1
    for l,data in enumerate(datasets):
        n = 1 - n
        if l % 2 == 0:
            m+=1

        dataset = Datos("Datasets/" + data)

        X = dataset.getDatos()[:,:-1] # Todos los atributos menos la clase
        y = dataset.getDatos()[:,-1] # Todas las clases

        for k,elem in enumerate(clase_atrib):
            print "-------------------ATRIB-------------------"

            iteracion = 1
            tasas_error = [0. for x in range(num_trees)]
            tasa_random_forest = 0.

            i = 0
            for train_index, test_index in skf.split(X, y):
                print "Iteracion " + str(i+1) + "/" + str(k_folds)

                X_train, y_train = X[train_index], y[train_index]
                Xt, yt = X[test_index], y[test_index]

                clf.fit(X_train, y_train)
                clf.predict_proba_error(Xt, class_atrib=elem)

                if k == 2:
                    clfRandom.fit(X_train, y_train)
                    tasa_random_forest += clfRandom.score(Xt, yt)

                for j in range(1,num_trees+1,2):
                    print "\tNumero arboles " + str(j) + "/100"
                    tasas_error[j-1] += 1 - clf.score_error(Xt, yt, n_classifiers=j, class_atrib=elem)

                i+=1

            if k == 2:
                tasa_random_forest /= k_folds
            error_final = list(map(lambda x: x/k_folds, tasas_error))

            print "Tasa de error: " + str(error_final)
            if k == 2:
                print "Tasa de error random forest: " + str(1 - tasa_random_forest)
            axs[m,n].plot(range(1,101,2),error_final[::2],linestyle=linestyle[k],color=color[k])
            if k == 2:
                axs[m,n].axhline(y=(1-tasa_random_forest), color='m', linestyle='--')
                #axs[m,n].plot(range(1,101,2),tasa_random_forest,linestyle='--',color='magenta')

        axs[m,n].legend(('0', '1', '0 - 1','Random forest'),loc='upper right')
        axs[m,n].set_title(data)
    plt.savefig("Imagenes/graficas_datasets_random_FINAL.eps")
    # plt.show()

except ValueError as e:
    print e
