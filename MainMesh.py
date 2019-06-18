# -*- coding: utf-8 -*-

from __future__ import division
from ClasificadorRuido import *
import numpy as np
import random
from plotModel import *
from sklearn.datasets import make_moons, make_circles, make_classification

import matplotlib.pyplot as plt

try:
    num_arboles = 1

    ###############################
    #########    MOONS    #########
    ###############################
    # X,y=make_moons(n_samples=500, shuffle=True, noise=0.2, random_state=None)
    # Xt,yt=make_moons(n_samples=100, shuffle=True, noise=0.2, random_state=None)

    ###############################
    ########    CIRCLES    ########
    ###############################
    X,y=make_circles(n_samples=500, noise=0.2, factor=0.2, random_state=None)
    Xt,yt=make_circles(n_samples=100, noise=0.2, factor=0.2, random_state=None)

    clf = ClasificadorRuido(num_arboles)
    clf.fit(X, y)

    print("Score: " + str(clf.score(Xt, yt)))
    print("Clasificaciones: " + str(clf.predict(Xt)))

    ####################################
    ##########     PLOT     ############
    ####################################

    print("------------PLOT------------")

    plotModel(Xt[:,0],Xt[:,1],yt,clf,1)

    #plt.show()
    # formato: conjuntodedatos_numarboles_<parametros_>_numeroejemplostest.eps
    # plt.show()
    plt.savefig("Imagenes/circles_1_02_probs_100.eps")

except ValueError as e:
    print(e)
