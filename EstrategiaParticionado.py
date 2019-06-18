from __future__ import division
from abc import ABCMeta,abstractmethod
from random import *
import numpy as np

import sys


class Particion:

  indicesTrain=[]
  indicesTest=[]

  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]

  def getTrain(self):
    return self.indicesTrain

  def getTest(self):
    return self.indicesTest

#####################################################################################################

class EstrategiaParticionado:

  # Clase abstracta
  __metaclass__ = ABCMeta

  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta
  nombreEstrategia="null"
  numeroParticiones=0
  particiones=[]

  @abstractmethod
  def creaParticiones(self,datos,seed=None):
    pass

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

  def __init__(self, numeroParticiones, porcentajeTrain):

    if porcentajeTrain < 0 or porcentajeTrain > 100:
      print("Error: el porcentaje ha de estar entre 0 y 100");
      sys.exit(0)

    self.nombreEstrategia = "ValidacionSimple"
    self.numeroParticiones = numeroParticiones
    self.porcentajeTrain = porcentajeTrain/100

  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
  # Devuelve una lista de particiones (clase Particion)
  def creaParticiones(self,datos,seed=None):
    self.particiones=[]

    numDatos = datos.getNumDatos()
    porcentaje = int(numDatos * self.porcentajeTrain)

    for i in range(self.numeroParticiones):
      arrayAleatorio = range(0, numDatos)
      shuffle(arrayAleatorio)

      particion = Particion()

      particion.indicesTrain = arrayAleatorio[0:porcentaje]
      particion.indicesTest = arrayAleatorio[porcentaje:]

      self.particiones.append(particion)

    return self.particiones


#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):


  def __init__(self, numeroParticiones):
    self.nombreEstrategia = "ValidacionCruzada"
    self.numeroParticiones = numeroParticiones

  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones
  # y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):
    self.particiones=[]

    numDatos = datos.getNumDatos()

    arrayAleatorio = range(0, numDatos)
    shuffle(arrayAleatorio)
    tupla = []

    if self.numeroParticiones > numDatos:
      self.numeroParticiones = numDatos
    elif self.numeroParticiones < 2:
      self.numeroParticiones = 2

    #1. Sacar num datos por cada particion
    numDatosParticion = int(numDatos / self.numeroParticiones)


    #2. Dividir en k particiones el total de datos
    for k in range(0, numDatos, numDatosParticion):
      if k + 2*numDatosParticion <= numDatos:
        tupla.append(arrayAleatorio[k:k+numDatosParticion])
      else:
        tupla.append(arrayAleatorio[k:])
        break

    #3. Bucle k veces generando k particiones
    for k in range(self.numeroParticiones):
      particion = Particion()
      particion.indicesTest = tupla[k]
      if k < self.numeroParticiones-1:
        particion.indicesTrain = np.hstack(tupla[:k] + tupla[k+1:])
      else:
        particion.indicesTrain = np.hstack(tupla[:k])

      self.particiones.append(particion)

    return self.particiones

#####################################################################################################

class ValidacionBootstrap(EstrategiaParticionado):


  def __init__(self, numeroParticiones):
    self.nombreEstrategia = "ValidacionBootstrap"
    self.numeroParticiones = numeroParticiones

  # Crea particiones segun el metodo de boostrap
  # Devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):
    self.particiones=[]

    numDatos = datos.getNumDatos()
    indexMax = numDatos-1

    listaIndex = range(0, indexMax)

    for i in range(self.numeroParticiones):
      particion = Particion()

      for j in range(numDatos):
        particion.indicesTrain.append(randint(0, indexMax))

      lista_aux = particion.indicesTrain

      particion.indicesTest = list(set(listaIndex)-set(lista_aux))

      self.particiones.append(particion)

    return self.particiones
