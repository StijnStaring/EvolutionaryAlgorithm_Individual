import Reporter
import numpy as np
from k_tournament import k_tournament
from copy import deepcopy
from random import shuffle, random, randint, choice
from representation import TravelingSalesPersonIndividual, TravelingSalesPersonProblem
from Nearest_Neighbor import Nearest_Neighbor
from three_opt import *
from elimination import elimination
from DPX import DPX
from SCX import SCX
# from DPX import DPX
# from inversion import inversion  # should be in the same file when submit


class r0620003:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    @staticmethod
    def initialize_population(problem:TravelingSalesPersonProblem, amount_of_cities_to_visit: int, initial_population_size: int) -> list:
        population: list = list()

        for _ in range(initial_population_size):

            indiv: TravelingSalesPersonIndividual = TravelingSalesPersonIndividual()
            candidate,cost_candidate = Nearest_Neighbor(amount_of_cities_to_visit, problem.weights)
            # candidate,cost_candidate = opt_3_local_search(problem, candidate,cost_candidate, max_iterations = 50) # how long have to search is a trade off between cost and profit
            indiv.set_order(candidate)
            indiv.set_cost(cost_candidate)
            # indiv.set_edges(edges_candidate)
            population.append(indiv)

        return population

    # The evolutionary algorithm's main loop
    def optimize(self, filename, initial_population_size: int = 100, k:int = 5, mutation_rate: float = 0.02, termination_value: int = 50):
        # def optimize(self, filename, initial_population_size: int = 100, amount_of_offsprings: int = 100, k: int = 5,alpha: float = .2, recombination_rate: float = 1, result_history_length: int = 50,termination_value: float = 0.005):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        amount_of_cities_to_visit = len(distanceMatrix)
        problem: TravelingSalesPersonProblem = TravelingSalesPersonProblem(distanceMatrix)
        population: list = self.initialize_population(problem,amount_of_cities_to_visit, initial_population_size)


        # StopingCriteria
        history_mean_objectives: list = []
        history_best_objectives: list = []
        count = 0
        result_history_length = 50
        bestScore_current = 0

        iteration = 1
        bestScore_individual: TravelingSalesPersonIndividual

        while count < termination_value:
            if iteration > 1:
                population.insert(0,bestScore_individual)

            offsprings: list = list()
            for _ in range(int(initial_population_size/2)):
                # Select best candidates
                parent_one: TravelingSalesPersonIndividual = k_tournament(population, k) # Can also try the selection with increasing exploitation --> decreasing s
                parent_two: TravelingSalesPersonIndividual = k_tournament(population, k)

                # Produce new offspring
                offspring = DPX(distanceMatrix,parent_one,parent_two)
                offsprings.append(offspring)
                # offspring_route1,offspring_cost1,offspring_edges1,offspring_route2,offspring_cost2,offspring_edges2 = DPX(problem, parent_one, parent_two)
                # offspring_route1, cost_offspring1 = opt_3_local_search(problem, offspring_route1, max_iterations=50)
                # offspring_route2, cost_offspring2 = opt_3_local_search(problem, offspring_route2, max_iterations=50)
                # offspring1 = TravelingSalesPersonIndividual()
                # offspring2 = TravelingSalesPersonIndividual()
                # offspring1.set_order(offspring_route1)
                # offspring2.set_order(offspring_route2)
                # offspring1.set_cost(offspring_route1)
                # offspring2.set_cost(offspring_route2)
                # offspring1.set_edges(offspring_edges1)
                # offspring2.set_edges(offspring_edges2)
                # offsprings.append(offspring1)
                # offsprings.append(offspring2)

            population += offsprings
            # population = [Greedy_Mutation(problem,indi) for indi in population]

            population = elimination(population, initial_population_size)

            if iteration > 1:
                bestScore_prev = history_best_objectives[-1]
            else:
                bestScore_prev = 0

            bestScore_individual = population[0]
            bestScore_current = bestScore_individual.get_cost()

            print("%s\t %s" % (np.around(bestScore_current,4),np.around(bestScore_prev,4)))

            if np.around(bestScore_current, 4) == np.around(bestScore_prev, 4):
                count += 1
            else:
                count = 0

            history_best_objectives.append(bestScore_current)
            scores = [x.get_cost() for x in population]
            meanScore = sum(scores) / len(scores)
            history_mean_objectives.append(meanScore)
            print("Mean objective: %s\tBest objective: %s\tBest individual %s" % (meanScore, bestScore_current, bestScore_individual))
            iteration += 1

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution

            timeLeft = self.reporter.report(meanScore, bestScore_current, np.array(bestScore_individual.get_order()))

            # Reduce memory
            if len(history_mean_objectives) > result_history_length:
                history_mean_objectives.pop(0)
                history_best_objectives.pop(0)

            if timeLeft < 0:
                break

        return bestScore_current


def run(args):
    (initial_population_size, k, mutation_rate, termination_value) = args
    instance = r0620003()
    print("Running...")
    result = instance.optimize("tour29.csv", initial_population_size=initial_population_size, k = k,mutation_rate = mutation_rate, termination_value=termination_value)
    print("Finished")
    return result


run((100, 1, 0.02, 100))
