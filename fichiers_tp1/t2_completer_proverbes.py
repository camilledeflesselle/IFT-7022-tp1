import json
import re
import operator
from nltk import word_tokenize
from nltk.util import pad_sequence, ngrams 
from nltk.lm.models import Laplace

proverbs_fn = "./data/proverbes.txt"
test1_fn = "./data/test_proverbes.txt"
BOS = '<BOS>'
EOS = '<EOS>'
models = dict()

def load_proverbs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()
    return [x.strip() for x in raw_lines]


def load_tests(filename):
    with open(filename, 'r') as fp:
        test_data = json.load(fp)
    return test_data

class ModelNgram:
    def __init__(self, text_list, **kwargs):
        """
        Initializer
        :param text_list: liste de textes utilisés pour construire le vocabulaire
        """
        self.text_list = text_list


    def build_vocabulary(self):
        all_unigrams = list()
        for sentence in self.text_list:
            word_list = word_tokenize(sentence.lower())
            all_unigrams = all_unigrams + word_list
        voc = set(all_unigrams)
        voc.add(BOS)
        voc.add(EOS)
        return list(voc)

    def get_ngrams(self, n):
        """
        :param n: taille des n-grammes
        :return: all_ngrams: les n-grammes construits
        """
        all_ngrams = list()
        for sentence in self.text_list:
            tokens = word_tokenize(sentence.lower())
            padded_sent = list(pad_sequence(tokens, pad_left=True, left_pad_symbol=BOS, pad_right=True, right_pad_symbol=EOS, n=n))
            all_ngrams = all_ngrams + list(ngrams(padded_sent, n=n))      
        return all_ngrams

    def get_model(self, n, vocabulary):
        """
        :param n: taille des n-grammes
        :param vocabulary: voabulaire utilisé
        :return model: le modèle associé, avec un lissage des probabilités
        """
        model = Laplace(n)
        corpus_ngrams = self.get_ngrams(n)
        model.fit([corpus_ngrams], vocabulary_text = vocabulary)
        return model
        

		
def train_models(filename):
    """
    Permet d'entraîner les différents modèles, enregistrés dans le dictionnaire 'models',
    qui permet de conserver les modèles de langue N-grammes après leur construction. 
    :param filename: nom du fichier où sont situés les proverbes
    """
    proverbs = load_proverbs(filename)
    model_init = ModelNgram(proverbs)
    vocabulary = model_init.build_vocabulary()
    # différents modèles
    for n in range(1, 4):  
        models[n] = model_init.get_model(n, vocabulary) 

def n_gram_proverb_test(tested_proverb, n):
    """
    :param tested_proverb: proverbe testé (utilisé pour le calcul de perplexité)
    :param n: longueur des n-grammes
    :return n_gram_tested: séquence de n-grammes associée au proverbe testé
    """
    tokens = word_tokenize(tested_proverb.lower())
    if n > 1 : n_gram_tested = list(ngrams(tokens, n=n))
    else : n_gram_tested = tokens
    return n_gram_tested


def cloze_test(incomplete_proverb, choices, n=3, criteria="perplexity"):
    """ 
    Fonction qui complète un texte à trous (des mots masqués) en ajoutant le bon mot.
    En anglais, on nomme ce type de tâche un "cloze test".
    :param criteria: indique la mesure qu'on utilise pour choisir le mot le plus probable: "logprob" ou "perplexity".
    :param n: le paramètre n désigne le modèle utilisé (1 - unigramme NLTK, 2 - bigramme NLTK, 3 - trigramme NLTK)
    :return 
            - result: proverbe complet estimé (c.-à-d. toute la séquence de mots du proverbe).
            - score: le score (logprob ou perplexité) associé
    """
    model = models[n]

    d_perplexity = dict()
    d_logscore = dict()
    for option in choices :
        tested_proverb = re.sub("\*{3}", option, incomplete_proverb)
        n_gram_tested = n_gram_proverb_test(tested_proverb, n)
        d_perplexity[tested_proverb] = model.perplexity(n_gram_tested)
        n_gram_context = re.sub("\*{3}", "", incomplete_proverb).split()
        d_logscore[tested_proverb] = model.logscore(option, n_gram_context)
    if criteria == "perplexity":
        result, score = min(d_perplexity.items(), key=operator.itemgetter(1))
    else:
        result, score = max(d_logscore.items(), key=operator.itemgetter(1))
    return result, score


if __name__ == '__main__':
    # Vous pouvez modifier cette section comme bon vous semble
    proverbs = load_proverbs(proverbs_fn)
    print("\nNombre de proverbes pour entraîner les modèles : ", len(proverbs))
    # Entraînement des modèles
    models = train_models(proverbs_fn)

    test_proverbs = load_tests(test1_fn)
    print("\nNombre de tests du fichier {}: {}\n".format(test1_fn, len(test_proverbs)))
    print("Les résultats des tests sont:")
    for n in range(1, 4):
        for criteria in ["perplexity", "logprob"]:
            nb_error = 0
            i = 0
            print("\n\n Résultats avec n = {} et le critère '{}' : ".format(n, criteria))
            for partial_proverb, options in test_proverbs.items():
                solution, valeur = cloze_test(partial_proverb, options, models, n, criteria=criteria)
                if solution not in proverbs : 
                    nb_error+=1
                    if nb_error < 5:
                        print("\n\tProverbe incomplet: {} , Options: {}".format(partial_proverb, options))
                        print("\tSolution = {} , Valeur = {}".format(solution, valeur))
                i+=1
            print("\n Nombre d'erreurs avec n = {} et le critère '{}' : {}".format(n, criteria, nb_error))
   
