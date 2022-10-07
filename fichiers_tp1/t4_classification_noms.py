# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import glob
import os
import string
from tkinter import WORD
import unicodedata
import json

import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd

datafiles = "./data/names/*.txt"  # les fichiers pour construire vos modèles
test_filename = './data/test_names.txt'  # le fichier contenant les données de test pour évaluer vos modèles

names_by_origin = {}  # un dictionnaire qui contient une liste de noms pour chaque langue d'origine
all_origins = []  # la liste des 18 langues d'origines de noms 
models = dict()
Vectorizers = dict()
N_MAX = 3

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
        names = [re.sub(r'^\s|\s$', '', name) for name in names] # supprime espaces en début et fin de noms
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
        character_ngrams_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, N_MAX))     
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
    
    for n in [str(i) for i in range(1,N_MAX+1)] + ['multi']:
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

def evaluate_train(classifier, n):
    # Evaluation par validation croisée du modèle sur les données d'entraînement
    X_train, y_train = data_and_labels(names_by_origin)
    X_train_vectorized = Vectorizers[str(n)+"-gram"].transform(X_train)
    accuracy_train = cross_val_score(classifier, X_train_vectorized, y_train, cv=5).mean()
    return accuracy_train
            

def plot_confusion_matrix(classifier, test_fn, type, n=3):
    font = {'size'   : 15}
    plt.rc('font', **font)
    test_data = load_test_names(test_fn)
    X_test, y_true = data_and_labels(test_data)
    y_pred = [origin(name, type, n) for name in X_test]
    fig = plt.figure(figsize = (15, 15))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(y_true, y_pred)
    cm = ConfusionMatrixDisplay(cm, display_labels = classifier.classes_)
    cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
    fig.delaxes(fig.axes[1]) #delete colorbar
    plt.xticks(rotation = 90)
    plt.xlabel('Classe prédite')
    plt.ylabel('Classe réelle')
    plt.savefig("../results_t4/confusion_matrix_{}_{}.png".format(type, n), bbox_inches = 'tight', dpi=500)
    

def get_words_with_highest_conditional_logprobabilities_by_class_NB(vectorizer, classifier):
    df_dict = dict()
    df = pd.DataFrame(vectorizer.get_feature_names(), columns =['N-grammes']) 
    for i in range(len(classifier.classes_)):
        c = classifier.classes_[i]
        df[c] = list(classifier.feature_log_prob_[i])
        df_dict[c] = df.sort_values(by=c, ascending=False)[0:10]
    return df_dict

def get_words_with_highest_conditional_logprobabilities_by_class_LR(vectorizer, classifier):
    df_dict = dict()
    df = pd.DataFrame(vectorizer.get_feature_names(), columns =['N-grammes']) 

    for i in range(len(classifier.classes_)):
        c = classifier.classes_[i]
        df[c] = list(classifier.coef_[i])
        df_dict[c] = list(df.sort_values(by=c, ascending=False)[0:10]['N-grammes'])
    return df_dict

if __name__ == '__main__':
    # Chargement des données d'entraînement et de test
    load_names()
    test_names = load_test_names(test_filename)
    total_names = 0

    # Nombre de noms par classe des données d'entraînement
    for origin_language, names in names_by_origin.items():
        total_names +=  len(names)
        print(" Nombre de noms d'origine {} : {} -> {}% des données".format(origin_language, len(names), round(len(names)/20074*100, 1)))
    print("Total names", total_names)

    # Entraînement de chacun des modèles
    train_classifiers()

    # Evaluation de chacun des modèles
    for model in ['NB', 'LR']:
        for ngram_length in [str(i) for i in range(1,N_MAX+1)]  + ['multi']:
            if ngram_length != 'multi' : ngram_length = int(ngram_length)
            classifier = get_classifier(model, n=ngram_length)

            accuracy_train = evaluate_train(classifier, ngram_length)
            print("\nAccuracy en entraînement pour {} / {}-gram = {}".format(model, ngram_length, accuracy_train))

            evaluation = evaluate_classifier(test_filename, model, n=ngram_length)
            print("\nAccuracy en test pour {} / {}-gram = {}".format(model, ngram_length, evaluation))

            # Affichage d'une matrice de confusions pour identifier les erreurs
            # plot_confusion_matrix(classifier, test_filename, model, n=ngram_length)
            
            # Affichage de l'importance des n-grammes pour l'un des modèles
            if model == 'LR' and ngram_length == 'multi' :
                df_dict = get_words_with_highest_conditional_logprobabilities_by_class_LR(Vectorizers[str(n)+"-gram"], classifier)
                print(df_dict)
