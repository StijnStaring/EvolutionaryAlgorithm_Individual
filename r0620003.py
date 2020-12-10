import numpy as np
from copy import deepcopy
from random import  random, randint, choice,shuffle
import time

class TravelingSalesPersonIndividual:
	def __init__(self):
		self.order = []
		self.cost = np.nan
		self.edges = []

	def get_order(self):
		return self.order

	def get_cost(self):
		return self.cost

	def get_edges(self):
		return self.edges

	def set_order(self, new_order):
		self.order = new_order

	def set_cost(self, new_cost):
		self.cost = new_cost

	def set_edges(self, new_edges):
		self.edges = new_edges

	def __str__(self) -> str:
		return str(self.get_order())

class TravelingSalesPersonProblem:

	def __init__(self, weights):
		self.weights = weights
		self.visited_vertices = {i for i in range(len(weights))}

	def get_weight(self, a, b):
		return self.weights[a][b]

	def get_dimension(self) -> int:
		return len(self.weights)

	def calculate_individual_score(self, individual: TravelingSalesPersonIndividual):
		order = individual.get_order()
		if len(order) == 0:
			return 0
		visited_edges = [(order[i], order[i + 1]) for i in range(0, len(order) - 1)]
		visited_edges += [(order[len(order) - 1], order[0])]

		path_weight = 0
		for (a, b) in visited_edges:
			path_weight += self.get_weight(a, b)

		return path_weight

def Nearest_Neighbor(amount_of_cities_to_visit, cost_matrix_original, random_numbers_to_start):
	if not random_numbers_to_start:
		NN_route = list(range(amount_of_cities_to_visit))
		shuffle(NN_route)
		return NN_route, []

	cost_matrix = deepcopy(cost_matrix_original)
	random_city = choice(random_numbers_to_start)
	random_numbers_to_start.remove(random_city)
	NN_route = [random_city]
	current_city = random_city

	while len(NN_route) != amount_of_cities_to_visit:
		costs = cost_matrix[current_city]
		costs[NN_route] = np.inf
		next_city = np.argmin(costs)
		NN_cost = costs[next_city]
		if NN_cost == np.inf:
			break
		NN_route.append(next_city)
		current_city = next_city

	NN_cost = cost_matrix_original[current_city][random_city]

	if NN_cost == np.inf or len(set(NN_route)) != amount_of_cities_to_visit:
		NN_route, random_numbers_to_start = Nearest_Neighbor(amount_of_cities_to_visit, cost_matrix_original,random_numbers_to_start)

	return NN_route, random_numbers_to_start

class Reporter:

	def __init__(self, filename):
		self.allowedTime = 300
		self.numIterations = 0
		self.filename = filename + ".csv"
		self.delimiter = ','
		self.startTime = time.time()
		self.writingTime = 0
		outFile = open(self.filename, "w")
		outFile.write("# Student number: " + filename + "\n")
		outFile.write("# Iteration, Elapsed time, Mean value, Best value, Cycle\n")
		outFile.close()

	# Append the reported mean objective value, best objective value, and the best tour
	# to the reporting file.
	#
	# Returns the time that is left in seconds as a floating-point number.
	def report(self, meanObjective, bestObjective, bestSolution):
		if time.time() - self.startTime < self.allowedTime + self.writingTime:
			start = time.time()

			outFile = open(self.filename, "a")
			outFile.write(str(self.numIterations) + self.delimiter)
			outFile.write(str(start - self.startTime - self.writingTime) + self.delimiter)
			outFile.write(str(meanObjective) + self.delimiter)
			outFile.write(str(bestObjective) + self.delimiter)
			for i in range(bestSolution.size):
				outFile.write(str(bestSolution[i]) + self.delimiter)
			outFile.write('\n')
			outFile.close()

			self.numIterations += 1
			self.writingTime += time.time() - start
		return (self.allowedTime + self.writingTime) - (time.time() - self.startTime)

def Opt_3(route_original:list,distanceMatrix,amount_cities,max_iterations=1):
	iteration = 0
	route = deepcopy(route_original)
	count = 0
	while count <= amount_cities and iteration < max_iterations:

		for i in range(amount_cities):

			""" Initialization of the current/next/previous cities """
			current_city = route[i]
			next_city:int
			next_city_i:int
			previous_city:int
			previous_city_i:int

			if i == 0:
				next_city = route[i + 1]
				next_city_i = i+1
				previous_city = route[amount_cities -1]
				previous_city_i = amount_cities - 1

			elif i == amount_cities -1:
				next_city = route[0]
				next_city_i = 0
				previous_city = route[i-1]
				previous_city_i = i - 1

			else:
				next_city = route[i+1]
				next_city_i = i + 1
				previous_city = route[i-1]
				previous_city_i = i - 1

			""" 3-opt forward case """
			costs1:list = distanceMatrix[current_city]
			current1:float = costs1[next_city]
			improvement1 = 0
			memory_cities1 = []

			candidates = [(c,co) for (c,co) in list(enumerate(costs1)) if co < current1 and c != previous_city and c != current_city]

			sorted_candidates_start = sorted(candidates, key=lambda x: x[1])

			for (c,co) in sorted_candidates_start:
				c_index = route.index(c)
				if c_index == amount_cities-1:
					city_D_i = 0
				else:
					city_D_i = c_index+1
				if c_index == 0:
					city_F_i = amount_cities -1
				else:
					city_F_i = c_index -1
				city_D = route[city_D_i]
				city_F = route[city_F_i]
				costs_c = deepcopy(distanceMatrix[c])
				costs1_relative = current1 + costs_c[city_D] - costs1[c]

				if previous_city_i >= c_index:
					not_desired_cities = [city_F,current_city, previous_city, next_city] + route[c_index:previous_city_i]
				else:
					not_desired_cities = [city_F,current_city, previous_city, next_city] + route[c_index:] + route[:previous_city_i]

				costs_c[not_desired_cities] = np.inf
				candidates2 = [(c2, co) for (c2, co) in list(enumerate(costs_c)) if co < costs1_relative]
				sorted_candidates_end = sorted(candidates2, key=lambda x: x[1])

				for (city_end,cost_end) in sorted_candidates_end:
					city_end_i = route.index(city_end)
					if city_end_i == len(route) - 1:
						city_end_next_i = 0
					else:
						city_end_next_i = city_end_i + 1

					city_end_next = route[city_end_next_i]

					cost_original = 0
					cost_reversed = 0

					insert_part: list
					if city_D_i >= city_end_next_i:
						insert_part = route[city_end_next_i:city_D_i]
					else:
						insert_part = route[city_end_next_i:] + route[:city_D_i]

					visited_edges = [(insert_part[i], insert_part[i + 1]) for i in range(0, len(insert_part) - 1)]
					for (a, b) in visited_edges:
						cost_original += distanceMatrix[a, b]

					insert_part.reverse()
					visited_edges = [(insert_part[i], insert_part[i + 1]) for i in range(0, len(insert_part) - 1)]
					for (a, b) in visited_edges:
						cost_reversed += distanceMatrix[a, b]

					improvement1 = costs1_relative + distanceMatrix[city_end][city_end_next] - distanceMatrix[city_end_next][next_city]  - distanceMatrix[city_end][city_D] + cost_original - cost_reversed
					memory_cities1 = [city_end_next_i, city_D_i,c,city_end]
					if improvement1 > 0:
						break
				if improvement1 > 0:
					break


			""" 3-opt backward case """

			costs1 = distanceMatrix[:,current_city]
			current1 = costs1[previous_city]
			improvement2 = 0
			memory_cities2 = []

			candidates = [(c, co) for (c, co) in list(enumerate(costs1)) if co < current1 and c != next_city and c != current_city]
			sorted_candidates_start = sorted(candidates, key=lambda x: x[1])

			for (c, co) in sorted_candidates_start:
				c_index = route.index(c)

				if c_index == amount_cities - 1:
					city_D_i = 0
				else:
					city_D_i = c_index + 1

				if c_index == 0:
					city_F_i = amount_cities - 1
				else:
					city_F_i = c_index - 1

				city_D = route[city_D_i]
				city_F = route[city_F_i]

				costs_c = deepcopy(distanceMatrix[:,c])
				costs1_relative = current1 + distanceMatrix[c,city_D] - distanceMatrix[c][current_city]

				if previous_city_i >= c_index:
					not_desired_cities = [city_F, current_city, previous_city, next_city] + route[c_index:previous_city_i]
				else:
					not_desired_cities = [city_F,current_city, previous_city, next_city] + route[c_index:] + route[:previous_city_i]

				costs_c[not_desired_cities] = np.inf
				candidates2 = [(c2, co) for (c2, co) in list(enumerate(costs_c)) if co < costs1_relative]
				sorted_candidates_end = sorted(candidates2, key=lambda x: x[1])

				for (city_end, cost_end) in sorted_candidates_end:
					city_end_i = route.index(city_end)
					if city_end_i == len(route) - 1:
						city_end_next_i = 0
					else:
						city_end_next_i = city_end_i + 1

					city_end_next = route[city_end_next_i]

					improvement2 = costs1_relative + distanceMatrix[city_end][city_end_next] - distanceMatrix[city_end][city_D] - distanceMatrix[previous_city][city_end_next]
					memory_cities2 = [city_end_next_i, city_D_i,c,city_end]

					if improvement2 > 0:
						break
				if improvement2 > 0:
					break

			""" Swapping best suggestion """
			if improvement1 <= 0 and improvement2 <= 0:
				count += 1
			else:
				count = 0

			if improvement1 >= improvement2 and improvement1 > 0:

				city_end_next_i = memory_cities1[0]
				city_D_i = memory_cities1[1]

				insert_part:list
				if city_D_i >= city_end_next_i:
					insert_part = route[city_end_next_i:city_D_i]
					insert_part.reverse()

				else:
					insert_part = route[city_end_next_i:] + route[:city_D_i]
					insert_part.reverse()

				if city_end_next_i>=next_city_i:
					part1 = route[next_city_i:city_end_next_i]
				else:
					part1 = route[next_city_i:] + route[:city_end_next_i]

				if previous_city_i >= city_D_i:
					part2 = route[city_D_i:previous_city_i+1]
				else:
					part2 = route[city_D_i:] + route[:previous_city_i+1]

				route = [current_city] + insert_part + part1 + part2

			elif improvement1 < improvement2 and improvement2 > 0:
				city_end_next_i = memory_cities2[0]
				city_D_i = memory_cities2[1]

				insert_part: list
				if city_D_i >= city_end_next_i:
					insert_part = route[city_end_next_i:city_D_i]

				else:
					insert_part = route[city_end_next_i:] + route[:city_D_i]

				if city_end_next_i >= next_city_i:
					part1 = route[next_city_i:city_end_next_i]
				else:
					part1 = route[next_city_i:] + route[:city_end_next_i]

				if previous_city_i >= city_D_i:
					part2 = route[city_D_i:previous_city_i+1]
				else:
					part2 = route[city_D_i:] + route[:previous_city_i+1]

				route = [current_city] + part1 + part2 + insert_part

		iteration += 1

	return route

def inversion(order: list, c:float,c_accent:float):
	index_c = order.index(c)
	index_c_accent = order.index(c_accent)

	if index_c < index_c_accent:
		part1 = order[:index_c+1]
		part2 = order[index_c+1:index_c_accent+1]
		part2.reverse()
		part3 = order[index_c_accent+1:]
		order[:] = part1 + part2 + part3 #

	elif index_c > index_c_accent:

		part1 = order[:index_c_accent + 1]
		part2 = order[index_c_accent + 1 : index_c + 1]
		part3 = order[index_c + 1 :]

		part_reversed = part3 + part1
		part_reversed.reverse()
		order[:] = part2 + part_reversed

	else:
		raise Exception("Should have been accepted by the previous two statements - inversion")

class r0620003:

	def __init__(self):
		self.reporter = Reporter(self.__class__.__name__)
	# 	self.reporter = Reporter.Reporter(self.__class__.__name__) --> Reporter class included in this file

	@staticmethod
	def initialize_population(problem: TravelingSalesPersonProblem, amount_of_cities_to_visit: int, initial_population_size: int, max_iterations:int) -> list:
		population: list = list()
		random_order: list = [i for i in range(1, amount_of_cities_to_visit)]
		random_numbers_to_start = deepcopy(random_order)
		random_numbers_to_start.insert(0, 0)
		for c in range(initial_population_size):
			indiv: TravelingSalesPersonIndividual = TravelingSalesPersonIndividual()
			candidate,random_numbers_to_start = Nearest_Neighbor(amount_of_cities_to_visit, problem.weights,random_numbers_to_start)
			candidate = Opt_3(candidate, problem.weights,amount_of_cities_to_visit,max_iterations)
			indiv.set_order(candidate)
			population.append(indiv)

		print('population initiated')
		return population

	def optimize(self, filename, initial_population_size: int = 100, p: float = 0.02, termination_value: float =  10, max_iterations: int = 1):

		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		timeLeft: float
		amount_of_cities_to_visit = len(distanceMatrix)
		problem: TravelingSalesPersonProblem = TravelingSalesPersonProblem(distanceMatrix)
		population: list = self.initialize_population(problem,amount_of_cities_to_visit, initial_population_size,max_iterations)

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
						copy_ind_order = Opt_3(copy_ind_order, problem.weights, amount_of_cities_to_visit, max_iterations)
						break

					inversion(copy_ind_order,c,c_accent) # order will be changed
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

			if bestScore_current == bestScore_prev:
				count += 1
			else:
				count = 0

			history_best_objectives.append(bestScore_current)
			meanScore = sum(scores) / len(scores)
			history_mean_objectives.append(meanScore)

			# Reduce memory
			if len(history_mean_objectives) > result_history_length:
				history_mean_objectives.pop(0)
				history_best_objectives.pop(0)
			print("It: %s\tMean objective: %s\tBest objective: %s\tBest individual %s" % (iteration,meanScore, bestScore_current,bestIndividual))

			iteration += 1

			timeLeft = self.reporter.report(meanScore, bestScore_current, np.array(bestIndividual))

			if timeLeft < 0:
				break

		return bestScore_current