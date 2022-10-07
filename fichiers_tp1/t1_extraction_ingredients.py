# -*- coding: utf-8 -*-
import re

ingredients_fn = "./data/ingredients.txt"
solutions_fn =  "./data/ingredients_solutions.txt"

# --> Expressions régulières utilisées
pattern_digit = r"[\d]+((,|\/|\sà\s)[\d]+)?"
pattern_unit_abrev = r"\b(o|c|m|l|k)+(l|L|b|z|g)\b|\bg\b"
pattern_unit_spoon = r"(c\.|cuillère)s?\sà\s\.?[A-zÀ-ÿ-é]+\.?"
pattern_unit_words_after_digit = r"(rôti\sde|tasse|pincée|gousse|Bouquet|Rondelle|enveloppe|tranche|botte|(bo.te?.\sde\s\S+))s?"
pattern_unit_words_without_digit = r"(.uel\S+\s\S+)|(\b\w+\sdemi)|(\bou\s|au\sg.*)"
pattern_quantity = re.compile('((\s\()?' +pattern_digit+ '(\s' +pattern_unit_spoon+ '|\s(' +pattern_unit_abrev+ '|' +pattern_unit_words_after_digit+ ')+)?\)?)+|' +pattern_unit_words_without_digit)
pattern_remove_first_space = "^\s"
pattern_remove_d = "(^\sd(e|'|’))"
pattern_complement = "\sen.*|\sr.pé|\spel\w+|\sém\w+|\scou.*|\shachées|\sbat.*|\stor.*|\sdan.*|\set\sd.*|\scis.*|\sd.en.*"
pattern_remove_after_comma = ",.*"

def load_ingredients(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_items = f.readlines()
    ingredients = [x.strip() for x in raw_items]
    return ingredients

def get_solutions(filename, colnames = ["ITEM", "QUANTITY", "INGREDIENT"]):
    solutions = load_ingredients(solutions_fn)
    quantities = []
    ingredients = []
    for solution in solutions :
        solution = solution.split(sep="   ")
        quantities.append(re.sub(r'QUANTITE:', "", solution[1]))
        ingredients.append(re.sub(r'INGREDIENT:', "", solution[2]))
    return ingredients, quantities

def get_ingredients(text):
    # Fonction qui permet de récupérer les ingrédients dans un texte
    # En entrée, on devrait recevoir une ligne de texte qui correspond à un ingrédient.
    # Par ex. 2 cuillères à café de poudre à pâte
    #
    x = re.search(pattern_quantity, text)
    if x :
        ingredient = re.sub(pattern_quantity, "", text)
        ingredient = re.sub(pattern_remove_d, "", ingredient)
        ingredient = re.sub(pattern_remove_after_comma, "", ingredient)
        ingredient = re.sub(pattern_remove_first_space, "", ingredient)
        ingredient = re.sub(pattern_complement, "", ingredient)
        
        quantity = x.group(0)

    else:
        ingredient = text
        quantity = ""
    return quantity, ingredient

if __name__ == '__main__':
    # Lecture des ingrédients du fichier
    print("Lecture des ingrédients du fichier {}. Voici quelques exemples: ".format(ingredients_fn))
    all_items = load_ingredients(ingredients_fn)

    # Lecture du fichier contenant les solutions pour chaque ligne
    ingredients, quantities = get_solutions(solutions_fn)

    # Représentation de quelques extractions fausses
    print("\nExtractions fausses :")
    print("Result / Truth")

    errors_quantities = []
    errors_ingredients = []

    for i in range(len(ingredients)):
        quantity, ingredient = get_ingredients(all_items[i])
        result = "\t{}\t QUANTITE: {}\t INGREDIENT: {}".format(all_items[i], quantity, ingredient)

        # On affiche les erreurs
        if quantity != quantities[i] :
            errors_quantities.append(result)
            print(quantity, "/", quantities[i])
        elif ingredient != ingredients[i] : 
            errors_ingredients.append(result)
            print(ingredient, "/", ingredients[i])

    print("Percentage of errors for quantities : {}%".format(len(errors_quantities)/len(quantities)*100))
    print("Percentage of errors for ingredients : {}%".format(len(errors_ingredients)/len(quantities)*100))

