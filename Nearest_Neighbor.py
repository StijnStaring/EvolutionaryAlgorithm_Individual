from representation import TravelingSalesPersonProblem
from random import randint
import numpy as np
from copy import deepcopy
from three_opt import opt_3_local_search

def Nearest_Neighbor(amount_of_cities_to_visit, cost_matrix_original):
    cost_matrix = deepcopy(cost_matrix_original)
    NN_route = []
    NN_edges = []
    random_city = randint(0,amount_of_cities_to_visit - 1)
    NN_route.append(random_city)
    current_city = random_city
    NN_cost = 0
    while len(NN_route) != amount_of_cities_to_visit:
        costs = cost_matrix[current_city]
        costs[NN_route] = np.inf # operation on the elements changes the original
        next_city = np.argmin(costs)
        NN_cost += costs[next_city]
        NN_edges.append((current_city,next_city))
        NN_route.append(next_city)
        current_city = next_city

    NN_cost += cost_matrix_original[current_city][random_city]
    NN_edges.append((current_city,random_city))

    return NN_route,NN_cost,NN_edges



def cost(problem: TravelingSalesPersonProblem,order:list):
    visited_edges = [(order[i], order[i + 1]) for i in range(0, len(order) - 1)]
    visited_edges += [(order[len(order) - 1], order[0])]

    path_weight = 0
    for (a,b) in visited_edges:
        print("(%s,%s) " % (a,b) )
        path_weight += problem.get_weight(a,b)

    return path_weight




def run(filename = "tour929.csv"):
    import numpy as np
    from random import shuffle
    try_list = list(range(929))
    shuffle(try_list)
    file = open(filename)
    distanceMatrix = np.loadtxt(file, delimiter=",")
    problem = TravelingSalesPersonProblem(distanceMatrix)
    file.close()
    all_costs = []
    print("Running...")
    for i in range(100):
        candidate,NN_cost,NN_edge = Nearest_Neighbor(len(try_list),distanceMatrix)
        # candidate,costss = opt_3_local_search(problem, candidate, max_iterations = 50)
        # all_costs.append(costss)
        # print("Candidate: %s ; NN_cost: %s ; NN_edge: %s ;" % (candidate,NN_cost,NN_edge))

    # print("The mean cost is: %s." % (sum(all_costs)/len(all_costs)))
    #     print(len(candidate))

    print("Finished")

run()
