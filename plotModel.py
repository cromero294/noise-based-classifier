from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from ClasificadorRuido import *

# Autor Luis Lago y Manuel Sanchez Montanes
# Modificada por Gonzalo
def plotModel(x,y,clase,clf,pred_prob=None):
    x_min, x_max = x.min() - .2, x.max() + .2
    y_min, y_max = y.min() - .2, y.max() + .2

    hx = (x_max - x_min)/100.
    hy = (y_max - y_min)/100.

    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

    if pred_prob == None:
        z1 = clf.predict(np.c_[xx.ravel(), yy.ravel()], 1)
        z2 = clf.predict(np.c_[xx.ravel(), yy.ravel()], 0)
        z3 = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        z1 = np.array(clf.predict_proba(np.c_[xx.ravel(), yy.ravel()], 1))[:, 1]
        z2 = np.array(clf.predict_proba(np.c_[xx.ravel(), yy.ravel()], 0))[:, 1]
        z3 = np.array(clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]))[:, 1]

    #z_list = [z1, z2, [], z1, z2, list(map(lambda x, y: x + y, z1, z2))] # Anyado z1 y z2 al final para poder pintarlas sin puntos en el mismo bucle
    z_list = [z1, z2, z3]

    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    fig.suptitle('Class distribution')

    for i in range(len(z_list)):
        z = np.array(z_list[i])
        z = z.reshape(xx.shape)
        # cm = plt.cm.RdBu
        cm = plt.cm.viridis
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        #ax = plt.subplot(1, 1, 1)
        axs[i].contourf(xx, yy, z, cmap=cm, alpha=.8)
        axs[i].contour(xx, yy, z, [0.5], linewidths=[2], colors=['k'])

        plt.gca().set_xlim(xx.min(), xx.max())
        plt.gca().set_ylim(yy.min(), yy.max())
        axs[i].grid(True)

    #plt.tight_layout()

    # for i in range(len(z_list)):
    #     plt.subplot(1, 3, i+1, squeeze=True)
    #
    #     z = np.array(z_list[i])
    #     z = z.reshape(xx.shape)
    #     cm = plt.cm.RdBu
    #     cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    #     #ax = plt.subplot(1, 1, 1)
    #     plt.contourf(xx, yy, z, cmap=cm, alpha=.8)
    #     plt.contour(xx, yy, z, [0.1], linewidths=[0.5], colors=['k'])
    #
    #     if clase is not None:
    #         plt.scatter(x[clase==0.], y[clase==0.], marker = 'o', c='red')
    #         plt.scatter(x[clase==1.], y[clase==1.], marker = 'o', c='blue')
    #     else:
    #         plt.plot(x,y,'g', linewidth=3)
    #
    #     plt.gca().set_xlim(xx.min(), xx.max())
    #     plt.gca().set_ylim(yy.min(), yy.max())
    #     #plt.grid(True)

    #plt.tight_layout()

def plotModel_arboles(x,y,clase,clf,axs,ind,pred_prob=None):
    x_min, x_max = x.min() - .2, x.max() + .2
    y_min, y_max = y.min() - .2, y.max() + .2

    hx = (x_max - x_min)/100.
    hy = (y_max - y_min)/100.

    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

    if pred_prob == None:
        if ind == 0:
            aux = clf.predict(np.c_[xx.ravel(), yy.ravel()], 0)
        elif ind == 1:
            aux = clf.predict(np.c_[xx.ravel(), yy.ravel()], 1)
        else:
            aux = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        if ind == 0:
            aux = np.array(clf.predict_proba(np.c_[xx.ravel(), yy.ravel()], 0))[:, 1]
        elif ind == 1:
            aux = np.array(clf.predict_proba(np.c_[xx.ravel(), yy.ravel()], 1))[:, 1]
        else:
            aux = np.array(clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]))[:, 1]

    z = np.array(aux)
    z = z.reshape(xx.shape)
    # cm = plt.cm.RdBu
    cm = plt.cm.viridis
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    im = axs.contourf(xx, yy, z, cmap=cm, alpha=.8)
    axs.contour(xx, yy, z, [0.5], linewidths=[2], colors=['k'])

    # if clase is not None:
    #     axs.scatter(x[clase==0.], y[clase==0.], marker = 'o', c='purple')
    #     axs.scatter(x[clase==1.], y[clase==1.], marker = 'o', c='yellow')

    plt.gca().set_xlim(xx.min(), xx.max())
    plt.gca().set_ylim(yy.min(), yy.max())
    axs.grid(True)

    return im
