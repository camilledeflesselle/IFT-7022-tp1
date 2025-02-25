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
from time import perf_counter
import pandas as pd

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
    
def lemmatization(X, eliminate_words = False, keeped_pos = ['ADJ']):
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
            if eliminate_words :
                if token.pos_ in keeped_pos:
                    review_lemmatized.append(token.lemma_)
            else :
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
    # le paramètre lowercase permet de convertir tous les caractères en minuscule avant la tokenisation
    # le paramètre stop_words = 'english' permet d'éliminer les stopwords dans la liste de tokens 
    # par exemple the, and, of, ... qui ne sont pas de bons indices
    vectorizer = CountVectorizer(analyzer = 'word', lowercase=True, stop_words = 'english', max_df = 0.7)
    vectorizer.fit(X_train)
    #print("\Nombre d'attributs de classification : ", len(vectorizer.get_feature_names()))

    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    if model == 'NB':
        # On construit un classificateur Naive Bayes sur les données d'entraînement
        # Multinomial = possiblement plusieurs classes
        classifier = MultinomialNB()
        classifier.fit(X_train_vectorized, y_train)
        important_words_by_class = get_words_with_highest_conditional_logprobabilities_by_class_NB(vectorizer, classifier)


    if model == 'LR':
        # On construit un classificateur Regression logistique sur les données d'entraînement
        classifier = LogisticRegression(C=0.1, penalty = 'l2', solver = 'liblinear')
        classifier.fit(X_train_vectorized, y_train)
        important_words_by_class = get_words_with_highest_conditional_logprobabilities_by_class_LR(vectorizer, classifier)

    accuracy_train = evaluation(classifier, X_train_vectorized, y_train, cross_val = True, i_val = 10)
    accuracy_test, confusion_matrix = evaluation(classifier, X_test_vectorized, y_test)
    
    # --> à décommenter si vous souhaitez connaître les mots les plus importants
    #print("Mots les plus importants pour la classe", POSITIVE)
    #print(important_words_by_class[POSITIVE])
    #print("Mots les plus importants pour la classe", NEGATIVE)
    #print(important_words_by_class[NEGATIVE])

    # Les résultats à retourner 
    results = dict()
    results['accuracy_train'] = accuracy_train
    results['accuracy_test'] = accuracy_test
    results['confusion_matrix'] = confusion_matrix  # la matrice de confusion obtenue de Scikit-learn
    return results

def get_words_with_highest_conditional_logprobabilities_by_class_NB(vectorizer, classifier):
    # Permet d'afficher les 10 n-grammes les plus importants pour un modèle de type NB
    #  - vectorizer : un objet CountVectorizer
    #  - classifier : un classifieur de type 'NB'
    df_dict = dict()
    df = pd.DataFrame(vectorizer.get_feature_names(), columns =['Mots']) 
    for i in range(len(classifier.classes_)):
            df[classifier.classes_[i]] = list(classifier.feature_log_prob_[i])
    label = "{} - {}".format(POSITIVE, NEGATIVE)
    df1 = df
    df1[label] = df[POSITIVE] - df[NEGATIVE]
    df1= df1.sort_values(by=POSITIVE, ascending=False)[0:200]

    df2 = df
    df2= df2.sort_values(by=NEGATIVE, ascending=False)[0:300]

    df_dict[POSITIVE] = list(df1.sort_values(by=label, ascending=False)[0:10]['Mots'])
    df_dict[NEGATIVE] = list(df2.sort_values(by=label, ascending=True)[0:10]['Mots'])
    return df_dict

def get_words_with_highest_conditional_logprobabilities_by_class_LR(vectorizer, classifier):
    # Permet d'afficher les 10 n-grammes les plus importants pour un modèle de type LR
    #  - vectorizer : un objet CountVectorizer
    #  - classifier : un classifieur de type 'LR'
    df_dict = dict()
    df = pd.DataFrame(vectorizer.get_feature_names(), columns =['Mots']) 
    df[classifier.classes_[1]] = list(classifier.coef_[0])
    df_dict[POSITIVE] = list(df.sort_values(by=POSITIVE, ascending=False)[0:10]['Mots'])
    df_dict[NEGATIVE] = list(df.sort_values(by=POSITIVE, ascending=True)[0:10]['Mots'])
    return df_dict


if __name__ == '__main__':
    # à changer en fonction de la configuration testée
    model = 'NB'
    normalization = 'words'
    
    # Entraînement et évaluation des modèles
    tps1 = perf_counter()
    results = train_and_test_classifier(reviews_dataset, model=model, normalization=normalization)
    tps2 = perf_counter()
    print("Résultats avec la méthode {} / {} :".format(model, normalization))
    print("Accuracy - entraînement: ", results['accuracy_train'])
    print("Accuracy - test: ", results['accuracy_test'])
    print("Matrice de confusion: ", results['confusion_matrix'])
    print("\nTemps d'exécution de train_and_test_classifier :", tps2-tps1)


