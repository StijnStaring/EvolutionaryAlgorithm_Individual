import numpy as np
from representation import TravelingSalesPersonProblem
from representation import TravelingSalesPersonIndividual
import random
from copy import deepcopy


def SCX(problem: TravelingSalesPersonProblem, individual1:TravelingSalesPersonIndividual, individual2: TravelingSalesPersonIndividual, recombination_rate: float):
    random_value: float = random.random()
    choose_a_parent_instead: bool = recombination_rate < random_value
    if choose_a_parent_instead:
        # order_for_copy_of_parent: list = None
        select_parent_one: bool = random_value >= .5
        if select_parent_one:
            return deepcopy(individual1)
        else:
            return deepcopy(individual2)
    
    dimension = problem.get_dimension()
    p1 = individual1.get_order() #list
    p2 = individual2.get_order() #list
    reference = np.arange(1,dimension) #start at node 0
    offspring = [0]


    while len(offspring) < dimension:

        index1 = p1.index(offspring[-1]) + 1
        if index1 >= dimension:
            index1 = 0

        index2 = p2.index(offspring[-1]) + 1
        if index2 >= dimension:
            index2 = 0

        cand1 = p1[index1]
        cand2 = p2[index2]

        cost1 = 0
        if cand1 not in offspring:
            cost1 = problem.get_weight(offspring[-1], cand1)
        else:
            for value  in reference:
                if value not in offspring:
                    cand1 = value
                    cost1 = problem.get_weight(offspring[-1], value)
                    break


        cost2 = 0
        if cand2 not in offspring:
            cost2 = problem.get_weight(offspring[-1], cand2)
        else:
            for value in reference:
                if value not in offspring:
                    cand2 = value
                    cost2 = problem.get_weight(offspring[-1], value)
                    break

        if cand1 == cand2:
            offspring.append(cand1)

        elif cost1 <= cost2:
            offspring.append(cand1)

        else:
            offspring.append(cand2)

    new_individual = TravelingSalesPersonIndividual()
    new_individual.set_order(offspring)
    return new_individual