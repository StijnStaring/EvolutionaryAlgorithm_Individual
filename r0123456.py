import Reporter
import numpy as np
from copy import deepcopy
from random import shuffle
from representation import TravelingSalesPersonIndividual, TravelingSalesPersonProblem
from k_tournament import k_tournament
from mutation import scrambleMutate, inverseMutate
from SCX import SCX
from elimination import elimination
from time import time
from multiprocessing import Pool, cpu_count

# Modify the class name to match your student number.
class r0123456:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	def initialize_population(self, amount_of_cities_to_visit: int, initial_population_size: int) -> list:
		return_value: list = list()
		random_order: list = [i for i in range(1, amount_of_cities_to_visit)]
		
		for _ in range(initial_population_size):
			shuffle(random_order)
			indiv: TravelingSalesPersonIndividual = TravelingSalesPersonIndividual()

			order_for_indiv: list = deepcopy(random_order)
			order_for_indiv.insert(0, 0)
			indiv.set_order(order_for_indiv)
			return_value.append(indiv)

		return return_value

	# The evolutionary algorithm's main loop
	def optimize(self, filename, initial_population_size: int = 100, amount_of_offsprings: int = 100, k: int = 5, alpha: float = .2, recombination_rate: float = 1, result_history_length: int = 50, termination_value: float =  0.005):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		# Your code here.
		population: list = self.initialize_population(len(distanceMatrix), initial_population_size)
		problem: TravelingSalesPersonProblem = TravelingSalesPersonProblem(distanceMatrix)
		history_mean_objectives: list = list()
		history_best_objectives: list = list()
		# bestObjective = 0

		while True:

			# Your code here.
			offsprings: list = list()
			for _ in range(amount_of_offsprings):
				parent_one: TravelingSalesPersonIndividual = k_tournament(problem, population, k)
				parent_two: TravelingSalesPersonIndividual = k_tournament(problem, population, k)

				offspring: TravelingSalesPersonIndividual = SCX(problem, parent_one, parent_two, recombination_rate)
				offsprings.append(offspring)
			
			population += offsprings
			population = [scrambleMutate(i, alpha) for i in population]
			population = elimination(problem, population, initial_population_size)

			bestSolution = np.array(population[0].get_order())
			scores = [problem.calculate_individual_score(indiv) for indiv in population]
			bestObjective = scores[0]
			meanObjective = sum(scores)/len(scores)

			print("Mean objective: %s\tBest objective: %s" % (meanObjective, bestObjective))
			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)

			history_mean_objectives.append(meanObjective)
			history_best_objectives.append(bestObjective)
			if len(history_mean_objectives) > result_history_length:
				history_mean_objectives.pop(0)
				history_best_objectives.pop(0)

			mean_mean: np.ndarray = np.mean(history_mean_objectives)
			best_mean: np.ndarray = np.mean(history_best_objectives)

			diff_means: float = abs(best_mean - mean_mean)
			if timeLeft < 0 or diff_means < termination_value:
				break

		# Your code here.
		return bestObjective


def run_once(args):
	(initial_population_size, amount_of_offsprings, k, alpha, recombination_rate) = args
	instance = r0123456()
	#print("Running...")
	result = instance.optimize("tour29.csv", initial_population_size=initial_population_size, amount_of_offsprings=amount_of_offsprings, k=k, alpha=alpha, recombination_rate=recombination_rate)
	return result


if __name__ == "__main__":
	print("Running this file on a PC with %s cores..." % (cpu_count()))

	poss_initial_population_sizes = [100, 200]
	poss_amount_of_offsprings = poss_initial_population_sizes
	poss_ks = [2, 5]
	poss_alphas = [.05, .3, 0.9]
	poss_recombination_rates = [.5, .8, 1]

	amount_of_possibilities: int = len(poss_initial_population_sizes) * len(poss_amount_of_offsprings) * len(poss_ks) * len(poss_alphas) * len(poss_recombination_rates)
	print("Found %s sets of parameters." % (amount_of_possibilities))

	for poss_initial_population_size in poss_initial_population_sizes:
		for poss_amount_of_offspring in poss_amount_of_offsprings:
			for poss_k in poss_ks:
				for poss_alpha in poss_alphas:
					for poss_recombination_rate in poss_recombination_rates:
						
						amount_of_iterations = 20
						p = Pool(processes=cpu_count())
						results = p.map(run_once, [(poss_initial_population_size, poss_amount_of_offspring, poss_k, poss_alpha, poss_recombination_rate) for i in range(amount_of_iterations)])
						
						#average_score = np.mean(results)
						best_score = min(results)
						avg_score = np.mean(results)

						print("#" * 50)
						print("Initial population size:			%s" % (poss_initial_population_size))
						print("Amount of offsprings:				%s" % (poss_amount_of_offspring))
						print("k (selection):					%s" % (poss_k))
						print("alpha (mutation):				%s" % (poss_alpha))
						print("Recombination rate:				%s" % (poss_recombination_rate))
						print("Best score over %s iterations:	%s" % (amount_of_iterations, best_score))
						print("Mean score over %s iterations:	%s" % (amount_of_iterations, avg_score))
						print(("#" * 50) + "\r\n")
