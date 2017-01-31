# Installation

TP SOUS LINUX !

Installez **conda** pour python 3:
```sh
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh ./Miniconda3-latest-Linux-x86_64.sh
```

Installez les dépendances suivantes:

```python
conda install numpy scipy mkl matplotlib
```

Installez la librairie **theano**:
```python
pip install --user Theano
```

Si vous voulez utiliser vos machines perso vous devrez vous débrouiller pour l'installation:

[**http://deeplearning.net/software/theano/install.html**](http://deeplearning.net/software/theano/install.html)

# Prise en main

Pour déveloper, vous pouvez utiliser l'éditeur Geany qui a un terminal intégré. Ecrivez votre code dans un fichier **main.py** que vous exécuterez avec la commande `python3 ./main.py`.

Commencez par importer **numpy** et **theano**:

```python
import numpy as np
import theano
import theano.tensor as tt
```

Theano étant basé sur le calcul symbolique, on peut distinguer trois types d'instructions:

1. déclarer une variable / expression symbolique ([`TensorVariable`](http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.TensorVariable)):

```python
x = tt.fscalar()
y = tt.fscalar()

z = x + y**2
```

2. compiler une expression symbolique en une fonction avec entrées / sorties ([`Function`](http://deeplearning.net/software/theano/library/compile/function.html#theano.compile.function.function)):

```python
f = theano.function(inputs=[x, y], outputs=z)
```

3. exécuter une fonction compilée:

```python
print(f(2, 3))
```

Notez que vous pouvez déclarer des variables symboliques munies d'un état interne persistant ([`TensorSharedVariable`](http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.TensorSharedVariable)):
```python
x = tt.fscalar()
a = theano.shared(value=1.0)

y = tt.power(x, a)
f = theano.function(inputs=[x], outputs=y)

print(f(2))

a.set_value(2.0)
print(f(2))
```

L'état interne d'une variable peut également être mis à jour lors de l'exécution d'une fonction compilée, en utilisant l'argument `updates`:
```python
x = tt.fscalar()
a = theano.shared(value=0.0)
b = theano.shared(value=0.0)

y = tt.power(x, a) + tt.power(x, b)
f = theano.function(inputs=[x], outputs=y, updates=[[a, a + 1.0], [b, b + 2.0]])

print(f(2))
print(f(2))
print(f(2))
```

Dans un réseau de neurones classique beaucoup d'opérations peuvent être vues comme des manipulations de scalaires, vecteurs (1D), matrices (2D) et tenseurs (nD). Par exemple, le code suivant constitue un modèle de type perceptron:
```python
x = tt.fvector()

w = theano.shared(value=np.asarray((1.2, 0.5, -0.2, 0.05, -1.1)))
b = theano.shared(value=0.1)

y = 1 / (1 + tt.exp(-(tt.dot(x, w) + b)))
```

Répondez aux questions suivantes:

+ Combien y a-t-il de variables en entrée dans ce modèle?
+ Combien de paramètres compte-t-on au total?
+ Qu'obtient-on en sortie de ce modèle lorsque toutes les entrées valent 1?

Notez que beaucoup de fonctions mathématiques sont déjà implémentées dans **theano**, comme la fonction sigmoïde avec `tt.nnet.sigmoid`. Dans l'exemple précédant, exprimez `y` en utilisant cette fonction, et vérifiez que vous obtenez bien la même chose.

Documentation utile:

+ [**types et fonctions de base**](http://deeplearning.net/software/theano/library/tensor/basic.html)
+ [**fonctions d'activation**](http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html)
+ [**evaluer une expression**](http://deeplearning.net/software/theano/library/compile/function.html)

# Regression logistique

Téléchargez la base MNIST [**ici**](http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz).

Récupérez le code de la régression logistique [**ici**](https://github.com/gasse/theano_tutorial/blob/master/logistic_regression.py).

Exécutez et étudiez ce code afin d'en comprendre les différentes étapes:

+ définition du modèle;
+ définition de deux fonctions coût (negative log-likelihood, hamming loss);
+ calcul du gradient et de la mise à jour des paramètres;
+ compilation des fonctions d'apprentissage et d'évaluation;
+ apprentissage et validation par mini-batches;
+ visualisation de 50 examples mal classés.

Vous pouvez jouer avec le nombre d'itérations (epochs) et le taux d'apprentissage (learning rate) pour obtenir différents résultats. Quel taux d'erreur arrivez-vous à atteindre avec ce modèle?

# Réseau de neurones

Modifiez le code de la régression logistique afin d'apporter les améliorations suivantes:

+ ajouter une (ou plusieurs) couches de neurones cachés, avec une fonction d'activation **tanh**;
+ ajouter un momentum à la descente de gradient pour accélérer l'apprentissage;
+ ajouter un terme de régularisation **L2** ou **L1** à la fontion coût (pénalisation des poids `w`) pour prévenir l'overfitting;
+ ajouter quelques couches de convolution avec max pooling ([`theano.tensor.nnet.conv2d`](http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv2d) et [`theano.tensor.signal.pool.pool2d`](http://deeplearning.net/software/theano/library/tensor/signal/pool.html#theano.tensor.signal.pool.pool_2d)).
