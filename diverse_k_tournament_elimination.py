from random import shuffle
from collections import deque
from copy import deepcopy
def diverse_k_tournament_elimination(population:list, k_elim: int, amount_of_survivors:int, amount_of_cities:int):
    copy_population = deepcopy(population)
    for indiv in population:
        order = indiv.get_order()
        index_c0 = order.index(0)
        items = deque(order)
        items.rotate(amount_of_cities - index_c0)
        indiv.set_order(list(items))

    survivors = []
    survivors_orders = []
    while len(survivors) != amount_of_survivors:
        if len(population) == 0:
            # raise Exception("The population is empty")
            shuffle(copy_population)
            chosen_individuals = copy_population[:k_elim]
            sorted_individuals = sorted(chosen_individuals, key=lambda individual: individual.get_cost())
            survivors.append(sorted_individuals[0])
        else:
            shuffle(population)
            chosen_individuals = list(enumerate(population[:k_elim]))
            sorted_individuals = sorted(chosen_individuals, key=lambda individual: individual[1].get_cost())
            candidate = sorted_individuals[0][1]
            if candidate.get_order() in survivors_orders:
                index_remove = sorted_individuals[0][0]
                del population[index_remove]

            else:
                survivors.append(candidate)
                survivors_orders.append(candidate.get_order()) # not good when work with 929
                index_remove = sorted_individuals[0][0]
                del population[index_remove]

    return survivors

