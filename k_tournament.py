from representation import TravelingSalesPersonProblem
from random import shuffle

def k_tournament(problem: TravelingSalesPersonProblem, list_of_individuals, k):
    shuffle(list_of_individuals)
    chosen_individuals = list_of_individuals[:k]
    sorted_individuals = sorted(chosen_individuals, key=lambda individual: problem.calculate_individual_score(individual))
    return sorted_individuals[0]
