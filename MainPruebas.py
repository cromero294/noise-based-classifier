from ClasificadorRuido import ClasificadorRuido
# from Datos import *
# import EstrategiaParticionado
# from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons, make_circles
# from sklearn.model_selection import StratifiedKFold

# from sklearn import tree
# import numpy as np
#
# import matplotlib.pyplot as plt
# import matplotlib

# # X,y=make_moons(n_samples=500, shuffle=True, noise=0.5, random_state=None)
# # Xt,yt=make_moons(n_samples=20000, shuffle=True, noise=0.5, random_state=None)
#
# X,y=make_circles(n_samples=500, shuffle=True, noise=0.2, random_state=None)
# Xt,yt=make_circles(n_samples=1000, shuffle=True, noise=0.2, random_state=None)
#
# # dataset=Datos('Datasets/example1.data')
# # estrategia = EstrategiaParticionado.ValidacionSimple(1, 95)
# # particiones = estrategia.creaParticiones(dataset)
# # datostrain = dataset.extraeDatos(particiones[0].getTrain())
# # datostest = dataset.extraeDatos(particiones[0].getTest())
#
# dataset = Datos("Datasets/bank-full.csv")
#
# X = dataset.getDatos()[:,:-1] # Todos los atributos menos la clase
# y = dataset.getDatos()[:,-1] # Todas las clases
#
# k_folds = 10
#
# skf = StratifiedKFold(n_splits=k_folds)
#
# clf = ClasificadorRuido()
# clfRandom = RandomForestClassifier(n_estimators=100)
#
# for train_index, test_index in skf.split(X, y):
#     X_train, y_train = X[train_index], y[train_index]
#     Xt, yt = X[test_index], y[test_index]
#
#     clfRandom.fit(X_train,y_train)
#     clf.fit(X_train, y_train)
#     clf.predict_proba_error(Xt)
#
#     print "Ceros: " + str(clf.score(Xt, yt, 0))
#     print "Unos: " + str(clf.score(Xt, yt, 1))
#     print "Score: " + str(clf.score(Xt, yt))
#     # print "Score error 1: " + str(clf.score_error(Xt, yt, 1))
#     print "Score error 100: " + str(clf.score_error(Xt, yt))
#     print "Random forest: " + str(clfRandom.score(Xt, yt))

# x1,x2,y = createDataSet(500,"twonorm",ymargin=0.0,noise=0.2,output_boundary=False)
# x1,x2,y = make_moons(n_samples=500, shuffle=True, noise=0.2, random_state=None)
# xt1,xt2,yt = createDataSet(500,"twonorm",ymargin=0.0,noise=0.2,output_boundary=False)
# x,y = make_moons(n_samples=500, shuffle=True, noise=0.2, random_state=None)
# x,y = make_circles(n_samples=500, noise=0.2, factor=0.2, random_state=None)

# X = np.c_[x1, x2]
# Xt = np.c_[xt1, xt2]

# colors = ['blue', 'red']

# print x

# plt.scatter(x[:,0], x[:,1], marker = 'o', c=np.array(y), cmap=matplotlib.colors.ListedColormap(colors))
# plt.scatter(np.array(xt1), np.array(xt2), marker = 'o', c=np.array(yt), cmap=matplotlib.colors.ListedColormap(colors))
# plt.show()
# plt.savefig("Imagenes/ejemplo_circles.png")

# clfRandom.fit(X,y)

X,y=make_moons(n_samples=100, shuffle=True, noise=0.5, random_state=None)
Xt,yt=make_moons(n_samples=200, shuffle=True, noise=0.5, random_state=None)

clf = ClasificadorRuido()

clf.fit(X, y)
clf.predict_proba_error(Xt)

# clf.predict(Xt)

# print "Ceros: " + str(clf.score(Xt, yt, 0))
# print "Unos: " + str(clf.score(Xt, yt, 1))
print "Score: " + str(clf.score(Xt, yt))
# # print "Score error 1: " + str(clf.score_error(Xt, yt, 1))
print "Score error 100: " + str(clf.score_error(Xt, yt))
# print "Random forest: " + str(clfRandom.score(Xt, yt))
