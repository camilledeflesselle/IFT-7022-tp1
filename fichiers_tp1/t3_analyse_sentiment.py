# -*- coding: utf-8 -*-
import json
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from nltk.stem.snowball import PorterStemmer
from nltk import word_tokenize
import spacy

reviews_dataset = {
    'train_pos_fn' : "./data/senti_train_positive.txt",
    'train_neg_fn' : "./data/senti_train_negative.txt",
    'test_pos_fn' : "./data/senti_test_positive.txt",
    'test_neg_fn' : "./data/senti_test_negative.txt"
}

POSITIVE = 'pos'
NEGATIVE = 'neg'

def load_reviews(filename):
    with open(filename, 'r') as fp:
        reviews_list = json.load(fp)
    return reviews_list

def train_test_from_files(dataset):
    """
    :param dataset: un dictionnaire contenant le nom des 4 fichiers utilisées pour entraîner 
                    et tester les classificateurs. Voir variable reviews_dataset.
    :return: 
            - X_train : les données d'entraînement
            - X_test : les données test
            - y_train : les labels associés à X_train
            - y_test : les labels associés à X_test
    """
    data = dict()
    for key, filename in dataset.items():
        var_name = re.sub("_fn", "", key)
        data[var_name] = load_reviews(filename)
    X_train = data['train_pos'] + data['train_neg']
    X_test = data['test_pos'] + data['test_neg']
    y_train = [POSITIVE] * len(data['train_pos']) + [NEGATIVE] * len(data['train_neg'])
    y_test = [POSITIVE] * len(data['test_pos']) + [NEGATIVE] * len(data['test_neg'])
    assert(len(X_train) == len(y_train))
    assert(len(X_test) == len(y_test))
    return X_train, X_test, y_train, y_test

def evaluation(classifier, X, y_true, cross_val = False, i_val = 10):
    """
    :param classifier: le classifier évalué
    :param X: stemmer utilisé pour réaliser le Stemming
    :param cross_val: un booléen valant True si l'on souhaite effectuer 
                      une validation croisée
    :param ival: le nombre d'itérations pour la validation croisée
    :return: 
            - accuracy: le pourcentage de biens classés
            - confusion_mat: la matrice de confusions (si cross_val égal à False)
    """
    if cross_val :
        scores = cross_val_score(classifier, X, y_true, cv=i_val)
        return scores.mean()
    else : 
        y_pred = classifier.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        confusion_mat = confusion_matrix(y_true, y_pred)
        return accuracy, confusion_mat
    
def lemmatization(X):
    """
    :param X: une liste de reviews
    :return: la liste de reviews normalisée avec la lemmatisation (spacy)
    """
    analyzer_en = spacy.load("en_core_web_sm")  
    X_lemmatized = []
    for review in X:
        review_lemmatized = []
        doc = analyzer_en(review)
        for token in doc:
            review_lemmatized.append(token.lemma_)
        X_lemmatized.append(' '.join(review_lemmatized))
    return X_lemmatized

def stemming(X, stemmer):
    """
    :param X: une liste de reviews
    :param stemmer: stemmer utilisé pour réaliser le Stemming
    :return: la liste de reviews normalisée avec le Stemming de Porter (nltk)
    """
    X_stemm = []
    for review in X:
        review_stemm = []
        tokens = word_tokenize(review)
        for token in tokens:
            stemm = stemmer.stem(token)
            review_stemm.append(stemm)
        X_stemm.append(' '.join(review_stemm))
    return X_stemm

def train_and_test_classifier(dataset, model='NB', normalization='words'):
    """
    :param dataset: un dictionnaire contenant le nom des 4 fichiers utilisées pour entraîner 
                    et tester les classificateurs. Voir variable reviews_dataset.
    :param model: le type de classificateur. NB = Naive Bayes, LR = Régression logistique.
    :param normalization: le prétraitement appliqué aux mots des critiques (reviews)
                 - 'word': les mots des textes sans normalization.
                 - 'stem': les racines des mots obtenues par stemming.
                 - 'lemma': les lemmes des mots obtenus par lemmatisation.
    :return: un dictionnaire contenant 3 valeurs:
                 - l'accuracy à l'entraînement (validation croisée)
                 - l'accuracy sur le jeu de test
                 - la matrice de confusion calculée par scikit-learn sur les données de test
    """

    # Récupération des données entraînement et tests, ainsi que les classes associées (pos ou neg)
    X_train, X_test, y_train, y_test = train_test_from_files(dataset)

    if normalization == 'lemma':
        X_train = lemmatization(X_train)
        X_test = lemmatization(X_test)
        
    if normalization == 'stem':
        porter_stemmer = PorterStemmer()
        X_train = stemming(X_train, porter_stemmer)
        X_test = stemming(X_test, porter_stemmer)

    # Le vectorizer permet de convertir les textes en sac de mots (vecteurs de compte)
    vectorizer = CountVectorizer(lowercase=True)
    vectorizer.fit(X_train)
    #print("\Nombre d'attributs de classification : ", len(vectorizer.get_feature_names()))

    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    if model == 'NB':
        # On construit un classificateur Naive Bayes sur les données d'entraînement
        # Multinomial = possiblement plusieurs classes
        classifier = MultinomialNB()
        classifier.fit(X_train_vectorized, y_train)

    if model == 'LR':
        # On construit un classificateur Regression logistique sur les données d'entraînement
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(X_train_vectorized, y_train)
    
    accuracy_train = evaluation(classifier, X_train_vectorized, y_train, cross_val = True, i_val = 10)
    accuracy_test, confusion_matrix = evaluation(classifier, X_test_vectorized, y_test)

    # Les résultats à retourner 
    results = dict()
    results['accuracy_train'] = accuracy_train
    results['accuracy_test'] = accuracy_test
    results['confusion_matrix'] = confusion_matrix  # la matrice de confusion obtenue de Scikit-learn
    return results


if __name__ == '__main__':
    # Vous pouvez modifier cette section comme vous le souhaitez.
    # Contenu des fichiers de données
    splits = ['train_pos_fn', 'train_neg_fn', 'test_pos_fn', 'test_neg_fn']
    print("Taille des partitions du jeu de données")
    partitions = dict()
    for split in splits:
        partitions[split] = load_reviews(reviews_dataset[split])
        print("\t{} : {}".format(split, len(partitions[split])))

    # Entraînement et évaluation des modèles
    results = train_and_test_classifier(reviews_dataset, model='LR', normalization='words')
    print("Accuracy - entraînement: ", results['accuracy_train'])
    print("Accuracy - test: ", results['accuracy_test'])
    print("Matrice de confusion: ", results['confusion_matrix'])

