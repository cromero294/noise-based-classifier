# Algoritmo de clasificación basado en la generación de ruido
# Noise generation based classifier

## Resumen
Dentro del machine learning, los problemas de clasificación siempre han constituido un principal
foco de atención. El objetivo de estos problemas es generar una regla a partir de un conjunto de datos
conocidos e intentar aplicar esa regla en un conjunto de datos desconocidos. Múltiples herramientas
se han construido para solucionar este tipo de problemas. Algunas de estas herramientas se han
combinado para abordar el problema desde otro punto de vista y han demostrado obtener resultados
significativamente mejores. Los objetivos principales de este trabajo son el desarrollo de un nuevo
algoritmo de clasificación basado en la combinación de otros clasificadores y en la generación de ruido
de los conjuntos de datos, y el análisis y estudio de este. El desarrollo de este modelo se basa en
la combinación de las decisiones que toman árboles de decisión entrenados con distintos conjuntos
de datos. Los conjuntos de datos con los que se entrenan los árboles de decisión son previamente
modificados mediante la generación de ruido. La generación de ruido que se realiza en este trabajo
consiste en la modificación de la etiqueta de ciertos datos del conjunto. Al entrenar cada árbol de
decisión con un conjunto diferente de datos, la decisión que va a tomar cada árbol es independiente
a la que toman el resto de los árboles. La combinación de estas decisiones hace que se consigan
buenos resultados. Los resultados obtenidos presentan una tasa de error menor que otros algoritmos
de clasificación similares como Random forest, también basado en conjuntos de clasificadores, en
algunos de los problemas propuestos. Los resultados que se adquieren dependen del contexto que se
aborde, aunque, de forma general, podemos afirmar que se obtiene una reducción de la tasa de error,
lo que supone que se ha desarrollado un algoritmo muy preciso.

## Abstract
Classification problems have always been a main focus of attention in machine learning. The goal
of these issues is to obtain a rule from a known data set in order to try to apply this rule in an unknown
data set. Several tools have been built to solve this kind of problems. Some of these tools have been
combined to address the problem from another point of view and the results obtained have been reported
to be significatively better. The main goals of this project are the development of a new classification
algorithm based on the combination of other classifiers and on the noise generation in the training data,
and the analysis and study of the algorithm created. The development of this model is based on the
combination of the decisions made by decision trees trained with different data sets. The data set used
to train this decision trees is previously modified by noise generation. The noise generation carried out
in this project consists in the label modification of some data set. Due to the fact that each decision tree
is trained with a different data set, the decision made by each tree does not depend on the decisions
made by the rest. Good results are acquired from the combination of these decisions. Our results show
a lower error rate in some of the proposed problems in comparison with other similar classification algorithms
such as Random forest, which is based on classifier ensemble as well. In conclusion, although
the results collected depend on the addressed context, in general, we can affirm that a very accurate
algorithm has been developed as we have managed to reduce the error rate.
