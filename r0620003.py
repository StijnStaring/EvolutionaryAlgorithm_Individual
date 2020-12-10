import Reporter
import numpy as np
# from k_tournament import k_tournament
from copy import deepcopy
from random import  random, randint, choice
from representation import TravelingSalesPersonIndividual, TravelingSalesPersonProblem
from Nearest_Neighbor import Nearest_Neighbor
from figure_layout import figure_layout
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
# from three_opt import *
# from elimination import elimination
# from DPX import DPX
# from collections import deque
# from Greedy_Mutation_2 import Greedy_Mutation_2
# from Greedy_Mutation import Greedy_Mutation
# from diverse_k_tournament_elimination import diverse_k_tournament_elimination
import time
from Opt_3_Swap import Opt_3
# from SCX import SCX
# from DPX import DPX
from inversion import inversion  # should be in the same file when submit
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
	def initialize_population(problem: TravelingSalesPersonProblem, amount_of_cities_to_visit: int, initial_population_size: int, max_iterations:int) -> list:
		# st = time.time()
		population: list = list()
		random_order: list = [i for i in range(1, amount_of_cities_to_visit)]
		random_numbers_to_start = deepcopy(random_order)
		random_numbers_to_start.insert(0, 0)
		for c in range(initial_population_size):
			# print("individuals added: %s " % len(population))

			indiv: TravelingSalesPersonIndividual = TravelingSalesPersonIndividual()

			if c < int(initial_population_size):
				# random_city = choice(random_numbers_to_start)
				# random_numbers_to_start.remove(random_city)
				# if c > 0:
				candidate,random_numbers_to_start = Nearest_Neighbor(amount_of_cities_to_visit, problem.weights,random_numbers_to_start)
				candidate = Opt_3(candidate, problem.weights,amount_of_cities_to_visit,max_iterations)  # how long have to search is a trade off between cost and profit
				indiv.set_order(candidate)
				# indiv.set_cost(cost_candidate)
				population.append(indiv)

				# else:
				# 	candidate = list(range(amount_of_cities_to_visit))
				# 	# cost_candidate = cost(problem, candidate)
				# 	candidate = Opt_3(candidate, problem.weights,amount_of_cities_to_visit) # how long have to search is a trade off between cost and profit
				# 	indiv.set_order(candidate)
				# 	# indiv.set_cost(cost_candidate)
				# 	population.append(indiv)


			# else:
			# 	shuffle(random_order)
			# 	order_for_indiv: list = deepcopy(random_order)
			# 	order_for_indiv.insert(0, 0)
			# 	# cost_for_indiv = cost(problem, order_for_indiv)
			# 	order_for_indiv = Opt_3(order_for_indiv, problem.weights, amount_of_cities_to_visit)



				# order_for_indiv, cost_for_indiv = Opt_3(order_for_indiv, cost_for_indiv, problem.weights,
				# 										amount_of_cities_to_visit)
				# indiv.set_order(order_for_indiv)
				# # indiv.set_cost(cost_for_indiv)
				# population.append(indiv)

		# print("random_numbers_to_start: %s" % random_numbers_to_start)
		# sorted_population = sorted(population, key=lambda individual: individual.get_cost())
		# # print('after NN best score: %s'% sorted_population[0].get_cost())
		# orders = [x.get_order() for x in sorted_population]
		# scores = [x.get_cost() for x in sorted_population]
		# print(scores)
		# print(orders)
		# et = time.time()
		# time_diff = et - st
		# print("Time needed for the initialization: %s " % time_diff)
		print('population initiated')
		return population

	# The evolutionary algorithm's main loop
	def optimize(self, filename, initial_population_size: int = 100, p: float = 0.02, termination_value: float =  10, max_iterations: int = 1):
	# def optimize(self, filename, initial_population_size: int = 100, amount_of_offsprings: int = 100, k: int = 5,alpha: float = .2, recombination_rate: float = 1, result_history_length: int = 50,termination_value: float = 0.005):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		timeLeft: float
		amount_of_cities_to_visit = len(distanceMatrix)

		problem: TravelingSalesPersonProblem = TravelingSalesPersonProblem(distanceMatrix)
		population: list = self.initialize_population(problem,amount_of_cities_to_visit, initial_population_size,max_iterations)


		# StopingCriteria
		history_mean_objectives: list = []
		history_best_objectives: list = []
		time_history: list = []
		count = 0
		result_history_length = 50
		# bestScore_current = 0

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
						copy_ind_order = Opt_3(copy_ind_order, problem.weights, amount_of_cities_to_visit, max_iterations)
						break

					# print("This is copy_ind_order before: %s." % copy_ind_order)
					inversion(copy_ind_order,c,c_accent) # order will be changed
					# print("This is copy_ind_order after: %s." % copy_ind_order)
					c = c_accent

				copy_ind.set_order(copy_ind_order)
				if problem.calculate_individual_score(copy_ind) < problem.calculate_individual_score(ind):
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

			if bestScore_current == bestScore_prev:
			# if np.around(bestScore_current,4) == np.around(bestScore_prev,4):
				count += 1
			else:
				count = 0

			history_best_objectives.append(bestScore_current)
			meanScore = sum(scores) / len(scores)
			history_mean_objectives.append(meanScore)
			print("It: %s\tMean objective: %s\tBest objective: %s\tBest individual %s" % (iteration,meanScore, bestScore_current,bestIndividual))

			iteration += 1


			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution

			timeLeft = self.reporter.report(meanScore, bestScore_current, np.array(bestIndividual))
			time_b = 300 - timeLeft
			time_history.append(time_b)

			# Reduce memory
			# if len(history_mean_objectives) > result_history_length:
			# 	history_mean_objectives.pop(0)
			# 	history_best_objectives.pop(0)

			if timeLeft < 0:
				break

		# time_b = 300 - timeLeft


		# return bestScore_current, meanScore
		return history_best_objectives,history_mean_objectives,time_history

def cost(problem: TravelingSalesPersonProblem,order:list):
	visited_edges = [(order[i], order[i + 1]) for i in range(0, len(order) - 1)]
	visited_edges += [(order[len(order) - 1], order[0])]

	path_weight = 0
	for (a,b) in visited_edges:
		path_weight += problem.get_weight(a,b)

	return path_weight


def run(args):
	(initial_population_size, p, termination_value,max_iterations) = args
	instance = r0620003()
	print("Running...")
	result = instance.optimize("tour29.csv", initial_population_size=initial_population_size, p = p, termination_value = termination_value,max_iterations = max_iterations)
	print("Finished")
	return result


# run((100,0.02,100,5)) # only three parameters makes the code more robust.

def convergence_plot(runs: int = 3, file = "tour929.csv", initial_population_size=50, p = 0.02, termination_value = 100,max_iterations = 1):
	colours = ['b','r','g']
	axis = figure_layout(titel="Convergence", xlabel="Time [s]",ylabel="Cost [-]")
	for i in range(runs):
		instance = r0620003()
		result = instance.optimize(file, initial_population_size=initial_population_size, p = p, termination_value = termination_value,max_iterations = max_iterations)
		axis.plot(result[2],result[0],linewidth = 2.0, color= colours[i])
		axis.plot(result[2],result[1],linestyle= 'dashed', linewidth=2.0, color=colours[i])

	plt.legend(['bestObjective','meanObjective','bestObjective','meanObjective','bestObjective','meanObjective'])
	plt.show()

convergence_plot()


# def histograms(runs: int = 3, file = "tour29.csv", initial_population_size=100, p = 0.02, termination_value = 50,max_iterations = 5):
# 	collection_best_score = []
# 	collection_mean_score = []
#
# 	for _ in range(1000):
# 		instance = r0620003()
# 		best,mean = instance.optimize(file, initial_population_size=initial_population_size, p=p,termination_value=termination_value, max_iterations=max_iterations)
# 		collection_best_score.append(best)
# 		collection_mean_score.append(mean)
#
# 	collection_best_score = np.array(collection_best_score)
# 	print("The mean of the best solutions is: %s \nThe standard deviation of the best solutions is: %s" % (np.mean(collection_best_score),np.std(collection_best_score)))
# 	collection_mean_score = np.array(collection_mean_score)
# 	print("The mean of the mean solutions is: %s \nThe standard deviation of the mean solutions is: %s" % (np.mean(collection_mean_score), np.std(collection_mean_score)))
#
# 	plt.figure(figsize=[10,8])
# 	n, bins, patches = plt.hist(x=collection_best_score, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
# 	plt.grid(axis='y', alpha=0.75)
# 	plt.xlabel('Value',fontsize=15)
# 	plt.xticks(fontsize=15)
# 	plt.yticks(fontsize=15)
# 	plt.ylabel('Frequency',fontsize=15)
# 	plt.title('Distribution of the best solutions',fontsize=15)
#
# 	plt.figure(figsize=[10, 8])
# 	n, bins, patches = plt.hist(x=collection_mean_score, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
# 	plt.grid(axis='y', alpha=0.75)
# 	plt.xlabel('Value', fontsize=15)
# 	plt.ylabel('Frequency', fontsize=15)
# 	plt.xticks(fontsize=15)
# 	plt.yticks(fontsize=15)
# 	plt.title('Distribution of the mean solutions', fontsize=15)
#
# 	plt.show()

# histograms()

def run_once(args):
	(initial_population_size, p, termination_value,max_iterations) = args
	instance = r0620003()
	#print("Running...")
	result = instance.optimize("tour29.csv", initial_population_size=initial_population_size, p=p, termination_value= termination_value, max_iterations = max_iterations)
	return result


if __name__ == "__main__":
	print("Running this file on a PC with %s cores..." % (cpu_count()))

	results_file_name = str(time.time()) + ".txt"
	results_file = open(results_file_name, "w")
	results_file.close()

	poss_initial_population_sizes = [20,75,100,150]
	poss_ps = [0.02, 0.15,0.25]
	poss_termination_values = [100]
	poss_Opt_3_iterations = [1,3,5]

	amount_of_possibilities: int = len(poss_initial_population_sizes) * len(poss_ps) * len(poss_termination_values) * len(poss_Opt_3_iterations)
	current_possibility_index: int = 1
	print("Found %s sets of parameters." % amount_of_possibilities)

	for poss_initial_population_size in poss_initial_population_sizes:
		for poss_p in poss_ps:
			for poss_Opt_3_iteration in poss_Opt_3_iterations:
				for poss_termination_value in poss_termination_values:
						amount_of_iterations = 8
						p = Pool(processes=cpu_count())
						results = p.map(run_once, [(poss_initial_population_size, poss_p, poss_termination_value, poss_Opt_3_iteration) for i in range(amount_of_iterations)])

						# average_score = np.mean(results)
						scores = [i[0] for i in results]
						times = [i[1] for i in results]
						best_score = min(scores)
						avg_score = np.mean(scores)
						avg_times = np.mean(times)

						results_file = open(results_file_name, "a")
						results_file.write("#" * 50 + "\r\n")
						results_file.write("Initial population size: %s" % poss_initial_population_size + "\r\n")
						# results_file.write("Amount of offsprings: %s" % poss_amount_of_offspring + "\r\n")
						# results_file.write("Amount of offsprings: %s" % poss_amount_of_offspring + "\r\n")
						# results_file.write("k (selection): %s" % poss_k + "\r\n")
						results_file.write("p (mutation): %s" % poss_p + "\r\n")
						results_file.write("poss_Opt_3_iteration: %s" % poss_Opt_3_iteration + "\r\n")
						# results_file.write("Recombination rate: %s" % poss_recombination_rate + "\r\n")
						results_file.write("# of termination iterations: %s" % poss_termination_value + "\r\n")
						results_file.write("Best score over %s iterations: %s" % (amount_of_iterations, best_score) + "\r\n")
						results_file.write("Mean score over %s iterations:	%s" % (amount_of_iterations, avg_score) + "\r\n")
						results_file.write("Mean time used:	%s  iterations: %s" % (amount_of_iterations, avg_times) + "\r\n")
						results_file.write(("#" * 50) + "\r\n")
						results_file.close()

						print("Completed %s / %s..." % (current_possibility_index, amount_of_possibilities))
						current_possibility_index += 1

	print("Finished simulation - data written to txt file.")