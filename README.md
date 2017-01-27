# Installation

TP SOUS LINUX !

Dans une console, installez la librairie **theano** pour python 3:

```python
pip3 install theano
```

Si vous voulez utiliser vos machines perso vous devrez vous débrouiller pour l'installation:

[**http://deeplearning.net/software/theano/install.html**](http://deeplearning.net/software/theano/install.html)

# Prise en main

Pour déveloper, vous pouvez utiliser l'éditeur Geany qui a un terminal intégré.

Commencer par importer **numpy** et **theano**:

```python
import numpy as np
import theano
import theano.tensor as tt
```

Theano étant basé sur le calcul symbolique, dans tout programme on peut distinguer trois étapes:

1. déclarer des variables symboliques;

```python
x = tt.dscalar()
y = tt.dscalar()
```

2. construire des expressions symboliques à partir de ces variables;

```python
z = x + y**2
```

3. évaluer une expression en donnant aux variables symboliques de vraies valeurs.

```python
f = theano.function(inputs=[x, y], outputs=z)
print(f(2, 3))
```

Il est également possible de déclarer des variables symboliques munies d'un état interne. Par exemple, essayez le code suivant:
```python
a = theano.shared(value=3.0)
y = tt.power(x, a)
f = theano.function(inputs=[x], outputs=y)

print(f(2))

a.set_value(2.0)
print(f(2))
```

Dans un réseau de neurones classique beaucoup d'opérations peuvent être vues comme des manipulations de scalaires, vecteurs (1D), matrices (2D) et tenseurs (nD). Par exemple, le code suivant constitue un modèle de type perceptron.
```python
x = tt.dvector()

w = theano.shared(value=np.asarray((1.2, 0.5, -0.2, 0.05, -1.1)))
b = theano.shared(value=np.asarray(0.1))

y = tt.nnet.sigmoid(tt.dot(x, w) + b)
```

Combien y a-t-il de variables en entrée dans ce modèle? Combien de paramètres compte-t-on au total? Evaluez ce modèle en mettant toutes les entrées à 1.

Documentation theano:

+ [**les tenseurs de base**](http://deeplearning.net/software/theano/library/tensor/basic.html)

