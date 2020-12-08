from representation import TravelingSalesPersonProblem
from random import shuffle,choice
import numpy as np
from copy import deepcopy
# from three_opt import opt_3_local_search

def Nearest_Neighbor(amount_of_cities_to_visit, cost_matrix_original,random_numbers_to_start):
    if not random_numbers_to_start:
        # print("start numbers are empty")
        NN_route = list(range(amount_of_cities_to_visit))
        shuffle(NN_route)
        return NN_route, []

    cost_matrix = deepcopy(cost_matrix_original)
    random_city = choice(random_numbers_to_start)
    random_numbers_to_start.remove(random_city)
    # NN_edges = []
    # random_city = randint(0,amount_of_cities_to_visit - 1)
    NN_route = [random_city]
    current_city = random_city
    # NN_cost = 0


    while len(NN_route) != amount_of_cities_to_visit:
        costs = cost_matrix[current_city]
        costs[NN_route] = np.inf # operation on the elements changes the original
        next_city = np.argmin(costs)
        NN_cost = costs[next_city]
        if NN_cost == np.inf:
            break
        # NN_edges.append((current_city,next_city))
        NN_route.append(next_city)
        current_city = next_city

    NN_cost = cost_matrix_original[current_city][random_city]

    if NN_cost == np.inf or len(set(NN_route)) != amount_of_cities_to_visit:
        NN_route,random_numbers_to_start = Nearest_Neighbor(amount_of_cities_to_visit,cost_matrix_original,random_numbers_to_start)
    # NN_edges.append((current_city,random_city))

    # if len(set(NN_route)) != amount_of_cities_to_visit: # better to not return anything and choose a different random city
    #     NN_route = list(range(amount_of_cities_to_visit))
    #     shuffle(NN_route)

    return NN_route,random_numbers_to_start
    # return NN_route,NN_cost



# def cost(problem: TravelingSalesPersonProblem,order:list):
#     visited_edges = [(order[i], order[i + 1]) for i in range(0, len(order) - 1)]
#     visited_edges += [(order[len(order) - 1], order[0])]
#
#     path_weight = 0
#     for (a,b) in visited_edges:
#         print("(%s,%s) " % (a,b) )
#         path_weight += problem.get_weight(a,b)
#
#     return path_weight


#
#
# def run(filename = "tour100.csv"):
#     import numpy as np
#     from random import shuffle
#
#     file = open(filename)
#     distanceMatrix = np.loadtxt(file, delimiter=",")
#     problem = TravelingSalesPersonProblem(distanceMatrix)
#     file.close()
#     all_costs = []
#     print("Running...")
#     for i in range(100):
#         candidate = Nearest_Neighbor(100,distanceMatrix)
#         candidate,costss = opt_3_local_search(problem, candidate,NN_cost, max_iterations = 15)
#         all_costs.append(costss)
#         # all_costs.append(NN_cost)
#         print("Candidate: %s ; NN_cost: %s ;" % (candidate,costss))
#         # print("Candidate: %s ; NN_cost: %s ;" % (candidate, NN_cost))
#         # print(len(candidate))
#         # print(len(set(candidate)))
#
#     print("The mean cost is: %s." % (sum(all_costs)/len(all_costs)))
#
#
#     print("Finished")
#
# run()