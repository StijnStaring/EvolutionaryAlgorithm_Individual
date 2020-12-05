import Reporter
import numpy as np
from copy import deepcopy
from random import shuffle,random,randint,choice
from representation import TravelingSalesPersonIndividual, TravelingSalesPersonProblem
from inversion import inversion # should be in the same file when submit
from Nearest_Neighbor import Nearest_Neighbor

class r0620003:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	# @staticmethod
	# def initialize_population(amount_of_cities_to_visit: int, initial_population_size: int) -> list:
	# 	return_value: list = list()
	# 	random_order: list = [i for i in range(1, amount_of_cities_to_visit)]
	#
	# 	for _ in range(initial_population_size):
	# 		shuffle(random_order)
	# 		indiv: TravelingSalesPersonIndividual = TravelingSalesPersonIndividual()
	#
	# 		order_for_indiv: list = deepcopy(random_order)
	# 		order_for_indiv.insert(0, 0)
	# 		indiv.set_order(order_for_indiv)
	# 		return_value.append(indiv)
	#
	# 	return return_value

	@staticmethod
	def initialize_population(problem: TravelingSalesPersonProblem, amount_of_cities_to_visit: int,initial_population_size: int) -> list:
		population: list = list()
		random_order: list = [i for i in range(1, amount_of_cities_to_visit)]
		for c in range(initial_population_size):

			indiv: TravelingSalesPersonIndividual = TravelingSalesPersonIndividual()
			if c < int(initial_population_size):
				candidate, cost_candidate = Nearest_Neighbor(amount_of_cities_to_visit, problem.weights)
				# candidate,cost_candidate = opt_3_local_search(problem, candidate,cost_candidate, max_iterations = 50) # how long have to search is a trade off between cost and profit
				indiv.set_order(candidate)
				indiv.set_cost(cost_candidate)
				# indiv.set_edges(edges_candidate)
				population.append(indiv)

			else:
				shuffle(random_order)
				order_for_indiv: list = deepcopy(random_order)
				order_for_indiv.insert(0, 0)
				indiv.set_order(order_for_indiv)
				cost_for_indiv = problem.calculate_individual_score(indiv)
				# candidate, cost_candidate = opt_3_local_search(problem, order_for_indiv, cost_for_indiv,max_iterations=10)  # how long have to search is a trade off between cost and profit
				# indiv.set_order(candidate)
				indiv.set_cost(cost_for_indiv)
				population.append(indiv)

		# sorted_population = sorted(population, key=lambda individual: individual.get_cost())
		# print('after NN best score: %s' % sorted_population[0].get_cost())
		# orders = [x.get_order() for x in sorted_population]
		# scores = [x.get_cost() for x in sorted_population]
		# print(scores)
		# print(orders)

		return population

	# The evolutionary algorithm's main loop
	def optimize(self, filename, initial_population_size: int = 100, p: float = 0.02, termination_value: float =  10):
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
		while count < termination_value:


			for i in range(0,len(population),1):
				ind = population[i]
				copy_ind: TravelingSalesPersonIndividual = deepcopy(ind)
				copy_ind_order = copy_ind.get_order()
				index = randint(0,amount_of_cities_to_visit - 1)
				index_next = 0
				index_prev = amount_of_cities_to_visit - 1
				if (index +1) < amount_of_cities_to_visit:
					index_next = index + 1
				if index > 0:
					index_prev = index - 1

				c_next = copy_ind_order[index_next]
				c_prev = copy_ind_order[index_prev]
				c = copy_ind_order[index]
				c_accent: int = 0

				while True:

					if random() <= p:
						index_l = randint(0, len(copy_ind_order)-1)
						while copy_ind_order[index_l] == c:
							index_l = randint(0, len(copy_ind_order) - 1)
						c_accent = copy_ind_order[index_l]

					else:
						selected_indiv = choice(population).get_order()
						index1 = selected_indiv.index(c) + 1
						if index1 >= amount_of_cities_to_visit:
							index1 = 0
						c_accent =  selected_indiv[index1]

					if c_accent == c_next or c_accent == c_prev:
						break

					# print("This is copy_ind_order before: %s." % copy_ind_order)
					inversion(copy_ind_order,c,c_accent) # order will be changed
					# print("This is copy_ind_order after: %s." % copy_ind_order)
					c = c_accent

				if problem.calculate_individual_score(copy_ind) <= problem.calculate_individual_score(ind):
					population[i] = copy_ind

			sorted_list_of_individuals = sorted(population,key=lambda individual: problem.calculate_individual_score(individual))
			bestIndividual: list = sorted_list_of_individuals[0].get_order()
			scores = [problem.calculate_individual_score(indiv) for indiv in sorted_list_of_individuals]

			if iteration > 1:
				bestScore_prev = history_best_objectives[-1]
			else:
				bestScore_prev = 0

			bestScore_current = scores[0]
			# print("%s\t %s" % (np.around(bestScore_current,4),np.around(bestScore_prev,4)))

			if np.around(bestScore_current,4) == np.around(bestScore_prev,4):
				count += 1
			else:
				count = 0

			history_best_objectives.append(bestScore_current)
			meanScore = sum(scores) / len(scores)
			history_mean_objectives.append(meanScore)
			print("Mean objective: %s\tBest objective: %s\tBest individual %s" % (meanScore, bestScore_current,bestIndividual))
			iteration += 1

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution

			timeLeft = self.reporter.report(meanScore, bestScore_current, np.array(bestIndividual))

			# Reduce memory
			if len(history_mean_objectives) > result_history_length:
				history_mean_objectives.pop(0)
				history_best_objectives.pop(0)

			if timeLeft < 0:
				break

		return bestScore_current


def run(args):
	(initial_population_size, p, termination_value) = args
	instance = r0620003()
	print("Running...")
	result = instance.optimize("tour929.csv", initial_population_size=initial_population_size, p = p, termination_value = termination_value)
	print("Finished")
	return result



run((200,0.02,10000))












# if __name__ == "__main__":
# 	print("Running this file on a PC with %s cores..." % (cpu_count()))
#
# 	poss_initial_population_sizes = [100, 200]
# 	poss_amount_of_offsprings = poss_initial_population_sizes
# 	poss_ks = [2, 5]
# 	poss_alphas = [.05, .3, 0.9]
# 	poss_recombination_rates = [.5, .8, 1]
#
# 	amount_of_possibilities: int = len(poss_initial_population_sizes) * len(poss_amount_of_offsprings) * len(poss_ks) * len(poss_alphas) * len(poss_recombination_rates)
# 	print("Found %s sets of parameters." % (amount_of_possibilities))
#
# 	for poss_initial_population_size in poss_initial_population_sizes:
# 		for poss_amount_of_offspring in poss_amount_of_offsprings:
# 			for poss_k in poss_ks:
# 				for poss_alpha in poss_alphas:
# 					for poss_recombination_rate in poss_recombination_rates:
#
# 						amount_of_iterations = 20
# 						p = Pool(processes=cpu_count())
# 						results = p.map(run_once, [(poss_initial_population_size, poss_amount_of_offspring, poss_k, poss_alpha, poss_recombination_rate) for i in range(amount_of_iterations)])
#
# 						#average_score = np.mean(results)
# 						best_score = min(results)
# 						avg_score = np.mean(results)
#
# 						print("#" * 50)
# 						print("Initial population size:			%s" % (poss_initial_population_size))
# 						print("Amount of offsprings:				%s" % (poss_amount_of_offspring))
# 						print("k (selection):					%s" % (poss_k))
# 						print("alpha (mutation):				%s" % (poss_alpha))
# 						print("Recombination rate:				%s" % (poss_recombination_rate))
# 						print("Best score over %s iterations:	%s" % (amount_of_iterations, best_score))
# 						print("Mean score over %s iterations:	%s" % (amount_of_iterations, avg_score))
# 						print(("#" * 50) + "\r\n")
