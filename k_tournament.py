from random import shuffle

def k_tournament(list_of_individuals, k):
    shuffle(list_of_individuals)
    chosen_individuals = list_of_individuals[:k]
    sorted_individuals = sorted(chosen_individuals, key=lambda individual: individual.get_cost())
    return sorted_individuals[0]
