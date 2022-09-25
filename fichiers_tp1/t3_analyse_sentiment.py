# -*- coding: utf-8 -*-
import json
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

reviews_dataset = {
    'train_pos_fn' : "./data/senti_train_positive.txt",
    'train_neg_fn' : "./data/senti_train_negative.txt",
    'test_pos_fn' : "./data/senti_test_positive.txt",
    'test_neg_fn' : "./data/senti_test_negative.txt"
}

POSITIVE = "pos"
NEGATIVE = "neg"

def load_reviews(filename):
    with open(filename, 'r') as fp:
        reviews_list = json.load(fp)
    return reviews_list

def train_test_from_files(dataset):
    train = []
    test = []
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

def evaluation(classifier, X, y_true):
    y_pred = classifier.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    confusion_mat = confusion_matrix(y_true, y_pred)
    return accuracy, confusion_mat
    
def train_and_test_classifier(dataset, model='NB', normalization='words'):
    """
    :param dataset: un dictionnaire contenant le nom des 4 fichiers utilisées pour entraîner et tester les classificateurs. Voir variable reviews_dataset.
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

    # Votre code...
    # Partionnement du corpus en données d'entraînement (train) et de test.
    # X désigne les textes à classifier et y les étiquettes associés à ces textes. 
    # Dans cet exemple, on garde 20% des questions pour les tests. 

    X_train, X_test, y_train, y_test = train_test_from_files(dataset)

    # Le vectorizer permet de convertir les textes en sac de mots (vecteurs de compte)
    vectorizer = CountVectorizer(lowercase=True)
    vectorizer.fit(X_train)
    print("\nNumbre d'attributs de classification : ", len(vectorizer.get_feature_names()))

    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    if model == 'NB':
        # On construit un classificateur Naive Bayes sur les données d'entraînement
        # Multinomial = possiblement plusieurs classes
        classifier = MultinomialNB()
        classifier.fit(X_train_vectorized, y_train)
        print("Type de classificateur : ", classifier)
        
        class_probs = list(zip(classifier.classes_, classifier.class_log_prior_))
        for x, prob in class_probs:
            print("logprob({}) = {}".format(x, round(prob,2)))
    
    accuracy_train, _ = evaluation(classifier, X_train_vectorized, y_train)
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
    results = train_and_test_classifier(reviews_dataset, model='NB', normalization='words')
    print("Accuracy - entraînement: ", results['accuracy_train'])
    print("Accuracy - test: ", results['accuracy_test'])
    print("Matrice de confusion: ", results['confusion_matrix'])

