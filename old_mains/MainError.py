# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:13:00 2018

@author: Mario Calle Romero
"""
from __future__ import division
from src.ClasificadorRuido import ClasificadorRuido
from sklearn.datasets import make_circles
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

try:
    # Xt,yt=make_moons(n_samples=20000, shuffle=True, noise=0.2, random_state=None)
    Xt,yt=make_circles(n_samples=20000, shuffle=True, noise=0.2, random_state=None)
    # Xt,yt=make_classification(n_samples=20000, shuffle=True, random_state=None)
    # datostest = np.column_stack((Xt, yt))

    num_trees = 101

    clf = ClasificadorRuido()
    clfRandom = RandomForestClassifier(n_estimators=num_trees)
    clase_atrib = [0, 1, None]
    linestyle = ['-.', ':', '-']
    color = ['red', 'olive', 'skyblue']

    for k,elem in enumerate(clase_atrib):
        print "-------------------ATRIB-------------------"

        iteracion = 1

        tasas_error = [0. for x in range(num_trees)]
        tasa_random_forest = 0.

        for i in range(100):
            print "Iteracion " + str(i+1) + "/100"

            # X,y=make_moons(n_samples=500, shuffle=True, noise=0.2, random_state=None)
            X,y=make_circles(n_samples=500, shuffle=True, noise=0.2, random_state=None)
            # X,y=make_classification(n_samples=500, shuffle=True, random_state=None)

            clf.fit(X, y)
            clf.predict_proba_error(Xt, class_atrib=elem)

            if k == 2:
                clfRandom.fit(X, y)
                tasa_random_forest += clfRandom.score(Xt, yt)

            for j in range(1,num_trees+1,2):
                print "\tNumero arboles " + str(j) + "/100"
                tasas_error[j-1] += 1 - clf.score_error(Xt, yt, n_classifiers=j, class_atrib=elem)

        if k == 2:
            tasa_random_forest /= 100
        error_final = list(map(lambda x: x/100, tasas_error))

        print "Tasa de error: " + str(error_final)
        if k == 2:
            print "Tasa de error random forest: " + str(1 - tasa_random_forest)
        plt.plot(range(1,102,2),error_final[::2],linestyle=linestyle[k],color=color[k])
        if k == 2:
            plt.axhline(y=(1-tasa_random_forest), color='m', linestyle='--')

    plt.legend(('0', '1', '0 - 1','Random forest'),loc='upper right')
    plt.title("Error - Clasificadores")
    # plt.show()
    plt.savefig("Imagenes/circles_error_FOREST_101_2.eps")

except ValueError as e:
    print e
