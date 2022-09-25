# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import glob
import os
import string
from tkinter import WORD
import unicodedata
import json

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

datafiles = "./data/names/*.txt"  # les fichiers pour construire vos modèles
test_filename = './data/test_names.txt'  # le fichier contenant les données de test pour évaluer vos modèles

names_by_origin = {}  # un dictionnaire qui contient une liste de noms pour chaque langue d'origine
all_origins = []  # la liste des 18 langues d'origines de noms 
models = dict()
X_train, y_train = list(), list()
Vectorizers = dict()

# Fonctions utilitaires pour lire les données d'entraînement et de test - NE PAS MODIFIER

def load_names():
    """Lecture des noms et langues d'origine d'un fichier. Par la suite,
       sauvegarde des noms pour chaque origine dans le dictionnaire names_by_origin."""
    for filename in find_files(datafiles):
        origin = get_origin_from_filename(filename)
        all_origins.append(origin)
        names = read_names(filename)
        names_by_origin[origin] = names
        

def find_files(path):
    """Retourne le nom des fichiers contenus dans un répertoire.
       glob fait le matching du nom de fichier avec un pattern - par ex. *.txt"""
    return glob.glob(path)


def get_origin_from_filename(filename):
    """Passe-passe qui retourne la langue d'origine d'un nom de fichier.
       Par ex. cette fonction retourne Arabic pour "./data/names/Arabic.txt". """
    return os.path.splitext(os.path.basename(filename))[0]


def read_names(filename):
    """Retourne une liste de tous les noms contenus dans un fichier."""
    with open(filename, encoding='utf-8') as f:
        names = f.read().strip().split('\n')
    return [unicode_to_ascii(name) for name in names]


def unicode_to_ascii(s):
    """Convertion des caractères spéciaux en ascii. Par exemple, Hélène devient Helene.
       Tiré d'un exemple de Pytorch. """
    all_letters = string.ascii_letters + " .,;'"
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def load_test_names(filename):
    """Retourne un dictionnaire contenant les données à utiliser pour évaluer vos modèles.
       Le dictionnaire contient une liste de noms (valeurs) et leur origine (clé)."""
    with open(filename, 'r') as fp:
        test_data = json.load(fp)
    return test_data

#---------------------------------------------------------------------------
# Fonctions à développer pour ce travail - Ne pas modifier les signatures et les valeurs de retour

def data_and_labels(names_with_origin):
    # Retourne la langue d'origine prédite pour le nom.
    #   - name = le nom qu'on veut classifier
    #   - type = 'NB' pour naive bayes ou 'LR' pour régression logistique
    #   - n désigne la longueur des N-grammes de caractères. Choix possible = 1, 2, 3, 'multi'
    #
    X = list()  # data
    y = list()  # labels
    for origin, names in names_with_origin.items():
        X = X + names
        y = y + [origin]*len(names)
    return X, y


def create_vectorizer(X_train, n=2):
    # Permet de créer un objet CountVectorizer, enregistré dans le dictionnaire Vectorizers
    # Permet de convertir les textes sous forme de sac de caractères
    #   - X_train = un vecteur contenant tous les noms d'entraînement pour créer les attributs
    #   - n désigne la longueur des N-grammes de caractères. Choix possible = 1, 2, 3, 'multi'
    #
    if n != 'multi' : 
        n = int(n)
        character_ngrams_vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n))  
    else : 
        character_ngrams_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 4))     
    character_ngrams_vectorizer.fit(X_train) 
    Vectorizers[str(n)+"-gram"] = character_ngrams_vectorizer


def train_classifiers():
    # Permet d'entraîner tous les classifieurs, pour les différents modèles et les différentes valeurs de n
    # Les classifieurs sont enregistrés dans le dictionnaire modèles
    # On suppose que les données d'entraînement ont été lues (load_names) et sont disponibles (names_by_origin).
    #
    # lecture des données d'entraînement et sont disponibles (names_by_origin).
    load_names()
    # données d'entraînement et classes associées
    X_train, y_train = data_and_labels(names_by_origin)
    
    for n in [str(i) for i in range(1,4)] + ['multi']:
        classifiers = dict()
        create_vectorizer(X_train, n)
        X_train_vectorized = Vectorizers[str(n)+"-gram"].transform(X_train)  
        classifiers['NB'] = MultinomialNB().fit(X_train_vectorized, y_train)
        classifiers['LR'] = LogisticRegression(max_iter=1000).fit(X_train_vectorized, y_train)
        models[str(n)+"-gram"] = classifiers
     
    
def get_classifier(type, n=3):
    # Retourne le classificateur correspondant. On peut appeler cette fonction
    # après que les modèles ont été entraînés avec la fonction train_classifiers
    #   - type = 'NB' pour naive bayes ou 'LR' pour régression logistique
    #   - n = 1,2,3 ou multi
    #
    key = str(n)+"-gram"
    if type == 'NB':
        classifier = models[key]['NB']
    elif type == 'LR':
        classifier = models[key]['LR']
    else:
        raise ValueError("Unknown model type")
    return classifier
    
    
def origin(name, type, n=3):
    # Retourne la langue d'origine prédite pour le nom.
    #   - name = le nom qu'on veut classifier
    #   - type = 'NB' pour naive bayes ou 'LR' pour régression logistique
    #   - n désigne la longueur des N-grammes de caractères. Choix possible = 1, 2, 3, 'multi'
    #
    key = str(n)+"-gram"
    name = Vectorizers[key].transform([name])  
    name_origin = models[key][type].predict(name)
    return name_origin 
    
    
def evaluate_classifier(test_fn, type, n=3):
    test_data = load_test_names(test_fn)
    X_test, y_true = data_and_labels(test_data)
    y_pred = [origin(name, type, n) for name in X_test]
    test_accuracy = accuracy_score(y_true, y_pred)
    return test_accuracy


if __name__ == '__main__':
    load_names()
    print("Les {} langues d'origine sont: \n{}".format(len(all_origins), all_origins))
    chinese_names = names_by_origin["Chinese"]
    print("\nQuelques noms chinois : \n", chinese_names[:20])

    train_classifiers()

    for model in ['NB', 'LR']:

        for ngram_length in [str(i) for i in range(1,4)]  + ['multi']:
            if ngram_length != 'multi' : ngram_length = int(ngram_length)
            classifier = get_classifier(model, n=ngram_length)
            #print("\nType de classificateur: ", classifier)

            #some_name = "Lamontagne"
            #some_origin = origin(some_name, model, n=ngram_length)
            #print("\nLangue d'origine de {}: {}".format(some_name, some_origin))

            test_names = load_test_names(test_filename)
            #print("\nLes données pour tester vos modèles sont:")
            #for org, name_list in test_names.items():
            #    print("\t{} : {}".format(org, name_list))
            evaluation = evaluate_classifier(test_filename, model, n=ngram_length)
            print("\nAccuracy pour {} / {}-gram = {}".format(model, ngram_length, evaluation))
