# -*- coding: utf-8 -*-
import re

ingredients_fn = "./data/ingredients.txt"
solutions_fn =  "./data/ingredients_solutions.txt"

# Mettre dans cette partie la (les) expression(s) régulière(s)
# que vous utilisez pour analyser les ingrédients
#
# Vos regex ici...
#
pattern_digit = "[\d]+((,|\/)[\d]+)?"
pattern_unit_abrev = "m|g|k|po|lb|l|L|oz"
pattern_unit_spoon = "(c\.|cuillère)s?\sà\s[A-zÀ-ü]+\.?"
pattern_unit_words = "(rôti\sde|tasse|pincée|gousses|Bouquet|Rondelle|enveloppe|tranche)s?"
pattern_quantity = "(\(?" + pattern_digit + "\s" +"(" +pattern_unit_spoon + "|" + pattern_unit_words + "|" + pattern_unit_abrev +")" + "(\.|é|\b)\)?)"
pattern_quantity = r"((\s\()?[\d]+((,|\/)[\d]+)?(\s(c\.|cuillère)s?\sà\s\.?[A-zÀ-ÿ-é]+\.?|\s((rôti\sde|tasse|Bouquet|Rondelle|pincée|gousse|enveloppe|tranche)s?|m|g|k|po|lb|l|L|oz)+\b)?\)?)+"

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
    # Insérez ici votre code pour l'extraction d'ingrédients.
    # En entrée, on devrait recevoir une ligne de texte qui correspond à un ingrédient.
    # Par ex. 2 cuillères à café de poudre à pâte
    # Vous pouvez ajouter autant de fonctions que vous le souhaitez.
    #
    # IMPORTANT : Ne pas modifier la signature de cette fonction
    #             afin de faciliter notre travail de correction.
    #
    # Votre code ici...
    #
    x = re.search(pattern_quantity, text)
    if x :
        ingredient = re.sub(pattern_quantity, "", text)
        # quantity = ' '.join(x.groups())
        quantity = x.group(0)

    else:
        ingredient = text
        quantity = ""
    return quantity, ingredient






if __name__ == '__main__':
    # Vous pouvez modifier cette section
    print("Lecture des ingrédients du fichier {}. Voici quelques exemples: ".format(ingredients_fn))
    all_items = load_ingredients(ingredients_fn)
    ingredients, quantities = get_solutions(solutions_fn)
    print("\nExtractions fausses :")
    print("Result / Truth")
    errors = []
    for i in range(len(ingredients)):
        quantity, ingredient = get_ingredients(all_items[i])
        result = "\t{}\t QUANTITE: {}\t INGREDIENT: {}".format(item, quantity, ingredient)
        if quantity != quantities[i] : 
            errors.append(result)
            print(quantity, "/", quantities[i])
    print("Percentage of errors : {}%".format(len(errors)/len(solutions)*100))
