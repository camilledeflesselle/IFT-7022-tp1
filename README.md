# IFT-7022-tp1

## Auteurs
- Lucas Chollet
- Camille Deflesselle

## Description des classes

### Classe BayesNaif

Cette classe est le classifieur implémenté dans le fichier *NaiveBayes.py*. Il s'agit de la classification bayésienne naïve. Son initialisation ne prend pas d'argument en entrée.


## Répartition des tâches de travail entre les membres d’équipe
Pour faciliter notre collaboration, nous avons créé un dépôt git privé, sur lequel se trouve tout notre travail.

Pour ce projet, l'un des membres de l'équipe a implémenté la classe KNN et l'autre la classe BayesNaif.
Quant aux fonctions dédiées au chargement des datasets, nous les avons écrit ensemble.

De même, nous avons implémenté une boucle d'entraînement/test (fichier *entrainer_tester.py*) en utilisant ces deux classes sur les trois jeux de données étudiés en travaillant ensemble sur le fichier. Ce fichier nous a permis de connaître les temps d'exécution des différents classifieurs (temps d'entraînement + évaluation sur les données test).
Dans la version actuelle de ce fichier, la recherche du meilleur k n'est pas faite. Pour la faire, il suffit de décommenter la ligne 54. Pour visualiser graphiquement cette recherche, il faut décommenter la ligne 55.
Par ailleurs, nous avions au préalable implémenté le fichier *metrics.py* qui nous permet d'afficher les différentes métriques de performances, que nous utilisons dans nos classes lors de l'évaluation.

## Explication des difficultés rencontrées dans ce travail

Globalement, pour ce travail tout s'est bien déroulé. Nos réflexions se sont essentiellement tournées vers le choix des hyperparamètres pour l'implémentation de l'algorithme KNN. Pour le premier jeu de données, iris dataset, qui ne contient que 150 instances, nous avons choisi une valeur de 5, ce qui engendre des échantillons de 30 instances. Prendre une valeur plus élevée ne nous semblait pas adaptée.

Aussi, la recherche du meilleur K pour les deux autres jeux de données utilisés prend un long temps d'exécution, ce qui nous laissait penser que notre code était incorrect. Finalement, nous comprenons que cela est normal, compte tenu du nombre d'instances et de la complexité en temps de l'algorithme. 

Finalement, nous pensons nous être bien approprié ces deux algorithmes d'apprentissage et avoir bien compris leur fonctionnement.
