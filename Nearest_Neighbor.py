from random import randint
import numpy as np
from copy import deepcopy

def Nearest_Neighbor(amount_of_cities_to_visit, cost_matrix_original):
    cost_matrix = deepcopy(cost_matrix_original)
    NN_route = []
    random_city = randint(0,amount_of_cities_to_visit - 1)
    NN_route.append(random_city)
    current_city = random_city

    while len(NN_route) != amount_of_cities_to_visit:
        costs = cost_matrix[current_city]
        costs[NN_route] = np.inf # operation on the elements changes the original
        next_city = np.argmin(costs)
        NN_route.append(next_city)
        current_city = next_city
    return NN_route


# def run(filename = "tour929.csv"):
#     import numpy as np
#     from random import shuffle
#     try_list = list(range(929))
#     shuffle(try_list)
#     file = open(filename)
#     distanceMatrix = np.loadtxt(file, delimiter=",")
#     file.close()
#     print("Running...")
#     for i in range(1):
#         candidate = Nearest_Neighbor(len(try_list),distanceMatrix)
#         print(candidate)
#
#     print("Finished")



run()



