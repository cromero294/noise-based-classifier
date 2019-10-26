#-*- coding: utf-8 -*-
import numpy as np
import re
import sys
from random import *

def createDataSet(n,model,ymargin=0.0,noise=None,output_boundary=False):
    x = np.random.rand(n,1)*2.0*np.pi
    xbnd = np.linspace(0,2.0*np.pi,100)

    if model == 'sine':
        y = (np.random.rand(n,1) - 0.5)*2.2
        c = y > np.sin(x)
        ybnd = np.sin(xbnd)
    elif model == 'linear':
        y = np.random.rand(n,1)*2.0*np.pi
        c = y > x
        ybnd = xbnd
    elif model == 'square':
        y = np.random.rand(n,1)*4.0*np.pi*np.pi
        c = y > x*x
        ybnd = xbnd*xbnd
    elif model == 'ringnorm':
        a = 2/np.sqrt(2.0)
        n2 = int(n/2)
        x[:n2] = np.random.normal(0,4.0,(n2,1))
        x[n2:] = np.random.normal(a,1.0,(n2,1))
        y = np.array(x)
        y[:n2] = np.random.normal(0,4.0,(n2,1))
        y[n2:] = np.random.normal(a,1.0,(n2,1))
        c = np.zeros_like(x)
        c[:n2]=1
        ybnd = xbnd*xbnd
    elif model == 'reringnorm':
        a = 2/np.sqrt(2.0)
        n2 = int(n/2)
        n4 = int(n/4)
        x[:n4] = np.random.normal(0,4.0,(n4,1))
        x[n4:3*n4] = np.random.normal(a,1.0,(n2,1))
        x[3*n4:] = np.random.normal(a,0.25,(n4,1))
        y = np.array(x)
        y[:n4] = np.random.normal(0,4.0,(n4,1))
        y[n4:3*n4] = np.random.normal(a,1.0,(n2,1))
        y[n2+n4:] = np.random.normal(a,0.25,(n4,1))
        c = np.zeros_like(x)
        c[n4:3*n4]=1
        ybnd = xbnd*xbnd
    elif model == 'xnorm':
        a = 2/np.sqrt(2.0)
        n2 = int(n/2)
        n4 = int(n/4)
        x[:n2] = np.random.normal(a,1.0,(n2,1))
        x[n2:] = np.random.normal(-a,1.0,(n2,1))
        y = np.array(x)
        y[:n4] = np.random.normal(a,1.0,(n4,1))
        y[n4:3*n4] = np.random.normal(-a,1.0,(n2,1))
        y[3*n4:] = np.random.normal(a,1.0,(n4,1))
        c = np.ones_like(x)
        c[:n4]=0
        c[n2:3*n4]=0
        ybnd = xbnd*xbnd
    elif model == 'threenorm':
        a = 2/np.sqrt(2.0)
        n2 = int(n/2)
        n4 = int(n/4)
        x[:n4] = np.random.normal(a,1.0,(n4,1))
        x[n4:n2] = np.random.normal(-a,1.0,(n4,1))
        x[n2:] = np.random.normal(a,1.0,(n2,1))
        y = np.array(x)
        y[:n4] = np.random.normal(a,1.0,(n4,1))
        y[n4:n2] = np.random.normal(-a,1.0,(n4,1))
        y[n2:] = np.random.normal(-a,1.0,(n2,1))
        c = np.ones_like(x)
        c[:n2]=0
        ybnd = xbnd*xbnd
    elif model == 'twonorm':
        a = 2/np.sqrt(2.0)
        n2 = int(n/2)
        x[:n2] = np.random.normal(a,1.0,(n2,1))
        x[n2:] = np.random.normal(-a,1.0,(n2,1))
        y = np.array(x)
        y[:n2] = np.random.normal(a,1.0,(n2,1))
        y[n2:] = np.random.normal(-a,1.0,(n2,1))
        c = np.ones_like(x)
        c[:n2]=0
        ybnd = xbnd*xbnd
    else:
        y = np.random.rand(n,1)*2.0*np.pi
        c = y > x
        ybnd = xbnd

    y[c == True] = y[c == True] + ymargin
    y[c == False] = y[c == False] - ymargin

    if noise is not None:
        y = y + noise * np.random.randn(n,1)
        x = x + noise * np.random.randn(n,1)

    if output_boundary == True:
        return np.matlib.matrix(x), np.matlib.matrix(y),np.matlib.matrix(c*1), xbnd, ybnd
    else:
        return np.matlib.matrix(x), np.matlib.matrix(y),np.matlib.matrix(c*1)

class Datos(object):

    TiposDeAtributos=('Continuo','Nominal')

    def __init__(self, nombreFichero):
        self.tipoAtributos = []
        self.nominalAtributos = []
        self.diccionarios = []

        listAux = []
        lista_de_listas = []

        try:
          fl = open (nombreFichero, "r")
          linea = fl.read().replace("\r", "").split("\n")
        except IOError:
          print ("Error: El archivo \"" + nombreFichero + "\" no se pudo abrir.")
          sys.exit(0)

        self.nDatos = int(linea[0])
        self.nombreAtributos = linea[1].split(",")

        #Configuracion tipoAtributos y nominalAtributos
        for word in linea[2].split(","):
          if word == "Continuo":
            self.tipoAtributos.append(Datos.TiposDeAtributos[0])
            self.nominalAtributos.append(False)
          elif word == "Nominal":
            self.tipoAtributos.append(Datos.TiposDeAtributos[1])
            self.nominalAtributos.append(True)
          else:
           raise ValueError('El tipo de dato \"' + word + '\" no es valido')

          self.diccionarios.append({})
          listAux.append([])

        #Creacion de listas con los distintos tipos de datos nominales y relleno de tuplas de datos
        for i in range(self.nDatos):
          datosAux = linea[i+3].split(",")
          lista_de_listas.append(datosAux)
          for j in range(len(datosAux)):
            if datosAux[j] not in listAux[j]:                                       #Si no estaba previamente
              if self.tipoAtributos[j] == Datos.TiposDeAtributos[1]:                #Si nominal
                listAux[j].append(datosAux[j])                                      #Agregamos

        #Configuracion de diccionarios discretizando los valores nominales
        for i in range(len(listAux)):
          listAux[i] = sorted(listAux[i])
          for j in range(len(listAux[i])):
            self.diccionarios[i].update({listAux[i][j]:len(self.diccionarios[i])})   #Agregamos y damos valor

        #Sustitucion de variables nominales por los valores de los diccionarios
        for i in range(len(lista_de_listas)):
          for j in range(len(self.diccionarios)):
            if self.nominalAtributos[j]:
              lista_de_listas[i][j] = float(self.diccionarios[j].get(lista_de_listas[i][j]))
            else:
              lista_de_listas[i][j] = float(lista_de_listas[i][j])

        self.datos = np.array(lista_de_listas)

        fl.close()

################################################################################
#                                                                              #
#                            getters y setters                                 #
#                                                                              #
################################################################################

    def getTipoAtributos(self):
        return self.tipoAtributos

    def getNombreAtributos(self):
        return self.nombreAtributos

    def getNominalAtributos(self):
        return self.nominalAtributos

    def getDiccionarios(self):
        return self.diccionarios

    def getDatos(self):
        return self.datos

    def getNumDatos(self):
        return self.nDatos

    def extraeDatos(self, idx):
        return self.datos[idx,:]
