import numpy as np
from representation import TravelingSalesPersonIndividual
from copy import deepcopy
from random import randint,choice,random
import numpy as np

def Greedy_Mutation_2(distanceMatrix, parent_one:TravelingSalesPersonIndividual, mutation_rate:float, amount_of_cities_to_visit:int):

    if random() > mutation_rate:
        # return parent_one
        order = parent_one.get_order()
        cost = parent_one.get_cost()
        return order,cost

    else:

        # Number of edges that are going to be removed is between 4 and 7.
        number = randint(4, 7)
        order = parent_one.get_order()
        cost1 = parent_one.get_cost()
        used_indices = []
        available_choice = list(range(amount_of_cities_to_visit))

        for _ in range(number):
            remove_edge_index = choice(available_choice)
            used_indices.append(remove_edge_index)
            available_choice.remove(remove_edge_index)

        used_indices = sorted(used_indices)

        old_cost = 0
        next_city:int
        for ui in used_indices:
            if ui != amount_of_cities_to_visit-1:
                next_city = order[ui + 1]
            else:
                next_city =0
            old_cost += distanceMatrix[order[ui]][next_city]

        parts = []
        for i in range(len(used_indices) - 1):
            parts.append(order[used_indices[i]:used_indices[i+1]])
        part1 = order[used_indices[-1]:] + order[:used_indices[0]]
        parts.insert(0,part1)


        build = parts[0]
        new_cost = 0
        for i in range(len(used_indices) - 1):
            rest = parts[i+1:]
            endpoints = list(build[-1]*np.ones(len(rest)))
            startpoints_rest = [s[0] for s in rest]
            costs_rest = list(map(get_cost_mutation,endpoints,startpoints_rest))
            chosen_part_index = np.argmin(costs_rest)
            new_cost = new_cost +  costs_rest[int(chosen_part_index)]
            build = build + rest[int(chosen_part_index)]

        new_cost += distanceMatrix[build[-1]][build[0]]


        return build,(cost1 + new_cost - old_cost)



def get_cost_mutation(a,b,distanceMatrix):
    return distanceMatrix[a,b]


