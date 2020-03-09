from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from src.ClasificadorRuido import *

# Autor Luis Lago y Manuel Sanchez Montanes
# Modificada por Gonzalo
def plot_model1(x,y,clase,clf,pred_prob=None):
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

def plot_model(clf, X, y, title):
    # Plotting decision regions
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = np.array(clf.predict_proba(np.c_[xx.ravel(), yy.ravel()]))[:, 1]
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z)

    aux = np.c_[X, y]
    print('hi')
    plt.scatter(X[np.where(aux[:, 2] == 0)[0][:50], 0], X[np.where(aux[:, 2] == 0)[0][:50], 1], c='purple', s=20, edgecolor='k', alpha=0.6)
    plt.scatter(X[np.where(aux[:, 2] == 1)[0][:50], 0], X[np.where(aux[:, 2] == 1)[0][:50], 1], c='blue', s=20, edgecolor='k', alpha=0.6)
    plt.scatter(X[np.where(aux[:, 2] == 2)[0][:50], 0], X[np.where(aux[:, 2] == 2)[0][:50], 1], c='yellow', s=20, edgecolor='k', alpha=0.6)

    plt.title("{}".format(title))

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
