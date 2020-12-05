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
from collections import deque
from Greedy_Mutation import Greedy_Mutation
from SCX import SCX
# from DPX import DPX
# from inversion import inversion  # should be in the same file when submit


class r0620003:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    @staticmethod
    def initialize_population(problem:TravelingSalesPersonProblem, amount_of_cities_to_visit: int, initial_population_size: int) -> list:
        population: list = list()
        random_order: list = [i for i in range(1, amount_of_cities_to_visit)]
        random_numbers_to_start = deepcopy(random_order)
        for c in range(initial_population_size):

            indiv: TravelingSalesPersonIndividual = TravelingSalesPersonIndividual()

            if c < int(initial_population_size / 2) and len(random_numbers_to_start) != 0:
                random_city = choice(random_numbers_to_start)
                random_numbers_to_start.remove(random_city)
                candidate,cost_candidate = Nearest_Neighbor(amount_of_cities_to_visit, problem.weights,random_city)
                # candidate,cost_candidate = opt_3_local_search(problem, candidate,cost_candidate, max_iterations = 50) # how long have to search is a trade off between cost and profit
                index_first_city = candidate.index(0)
                amount_to_move = amount_of_cities_to_visit - index_first_city
                items = deque(candidate)
                items.rotate(amount_to_move)

                if list(items) not in population:
                    indiv.set_order(list(items))
                    indiv.set_cost(cost_candidate)
                    # indiv.set_edges(edges_candidate)
                    population.append(indiv)

                else:
                    shuffle(random_order)
                    order_for_indiv: list = deepcopy(random_order)
                    order_for_indiv.insert(0, 0)
                    indiv.set_order(order_for_indiv)
                    cost_for_indiv = problem.calculate_individual_score(indiv)
                    # candidate,cost_candidate = opt_3_local_search(problem, order_for_indiv,cost_for_indiv, max_iterations = 10) # how long have to search is a trade off between cost and profit
                    # indiv.set_order(candidate)
                    indiv.set_cost(cost_for_indiv)
                    population.append(indiv)

            else:
                shuffle(random_order)
                order_for_indiv: list = deepcopy(random_order)
                order_for_indiv.insert(0, 0)
                indiv.set_order(order_for_indiv)
                cost_for_indiv = problem.calculate_individual_score(indiv)
                # candidate,cost_candidate = opt_3_local_search(problem, order_for_indiv,cost_for_indiv, max_iterations = 10) # how long have to search is a trade off between cost and profit
                # indiv.set_order(candidate)
                indiv.set_cost(cost_for_indiv)
                population.append(indiv)

        sorted_population = sorted(population, key=lambda individual: individual.get_cost())
        print('after NN best score: %s'% sorted_population[0].get_cost())
        orders = [x.get_order() for x in sorted_population]
        scores = [x.get_cost() for x in sorted_population]
        print(scores)
        print(orders)

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
        bestScore_individual = TravelingSalesPersonIndividual()

        while count < termination_value:
            k = get_the_k_value(iteration)
            if iteration > 1:
                population.insert(0,bestScore_individual) # elitism --> insert the best individual

            offsprings: list = list()
            for _ in range(initial_population_size):
            # for _ in range(int(10)):
                # Select best candidates
                parent_one: TravelingSalesPersonIndividual = k_tournament(population, k) # Can also try the selection with increasing exploitation --> decreasing s
                parent_two: TravelingSalesPersonIndividual = k_tournament(population, k)

                # Produce new offspring
                offspring = DPX(distanceMatrix,parent_one,parent_two)
                offsprings.append(offspring)

            population += offsprings
            population = [Greedy_Mutation(distanceMatrix, individual) for individual in population]
            k_elim = 3
            population = shared_fitness_k_tournament_elimination(problem, population, initial_population_size,k_elim)


                # population[randint(0,amount_of_cities_to_visit - 1)] = offspring
            #     not good elimination!!! 

            # population = offsprings
            # population = [Greedy_Mutation(problem,indi) for indi in population]

            # population = elimination(population, initial_population_size)

            if iteration > 1:
                bestScore_prev = history_best_objectives[-1]
            else:
                bestScore_prev = 0

            sorted_population = sorted(population, key=lambda individual: individual.get_cost())
            bestScore_individual = sorted_population[0]
            # bestScore_individual = population[0]
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

def get_the_k_value(iteration:int) -> int:
    if iteration > 10:
        return 1
    elif iteration > 30:
        return 2
    elif iteration > 60:
        return 3
    elif iteration > 90:
        return 4
    elif iteration > 120:
        return 5
    else:
        raise Exception("Should have been excepted by the previous cases. ")


def run(args):
    (initial_population_size, k, mutation_rate, termination_value) = args
    instance = r0620003()
    print("Running...")
    result = instance.optimize("tour29.csv", initial_population_size=initial_population_size, k = k,mutation_rate = mutation_rate, termination_value=termination_value)
    print("Finished")
    return result


run((100, 5, 0.02, 100))
