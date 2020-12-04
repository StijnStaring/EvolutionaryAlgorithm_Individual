import numpy as np
from representation import TravelingSalesPersonIndividual
from copy import deepcopy
from random import randint

def DPX(cost_matrix_original, parent_one:TravelingSalesPersonIndividual, parent_two:TravelingSalesPersonIndividual):
    order1 = parent_one.get_order()
    order2 = parent_two.get_order()

    visited_edges1 = [(order1[i], order1[i + 1]) for i in range(0, len(order1) - 1)]
    visited_edges1 += [(order1[len(order1) - 1], order1[0])]

    visited_edges2 = [(order2[i], order2[i + 1]) for i in range(0, len(order2) - 1)]
    visited_edges2 += [(order2[len(order2) - 1], order2[0])]

#     Look for the edges in common.
    inters = list(set(visited_edges1).intersection(set(visited_edges2)))
#     inters = [(918, 323), (685, 606),(700,701)]
#     print("inters is: %s " % inters)

#     Build the offspring
    cost_matrix = deepcopy(cost_matrix_original)
    amount_of_cities_to_visit = len(cost_matrix)
    NN_route = []
    # NN_edges = []
    random_city = randint(0, amount_of_cities_to_visit - 1)
    NN_route.append(random_city)
    current_city = random_city
    NN_cost = 0
    y = []
    while len(NN_route) != amount_of_cities_to_visit:
        if len(inters) != 0:
            y = list(filter(lambda x: fun(current_city,x),inters))


        costs = cost_matrix[current_city]
        if len(inters) != 0:
            for (c1,c2) in inters:
                costs[c2] = np.inf
        if len(y) != 0:
            next_city = y[0][1]
            NN_cost += costs[next_city]
            # NN_edges.append((current_city,next_city))
            NN_route.append(next_city)
            current_city = next_city

        else:
            costs[NN_route] = np.inf  # operation on the elements changes the original
            next_city = np.argmin(costs)
            NN_cost += costs[next_city]
            # NN_edges.append((current_city,next_city))
            NN_route.append(next_city)
            current_city = next_city

    NN_cost += cost_matrix_original[current_city][random_city]
    # NN_edges.append((current_city,random_city))

    offspring = TravelingSalesPersonIndividual()
    offspring.set_order(NN_route)
    offspring.set_cost(NN_cost)

    return offspring

def fun(current_city,list_edges):

    (sc,ec) =  list_edges
    if current_city == sc:
        return True

    else:
        return False


# def run(filename = "tour929.csv"):
#     import numpy as np
#     from random import shuffle
#     from representation import TravelingSalesPersonIndividual, TravelingSalesPersonProblem
#     try_list1 = list(range(929))
#     try_list2 = list(range(929))
#     shuffle(try_list1)
#     shuffle(try_list2)
#     file = open(filename)
#     distanceMatrix = np.loadtxt(file, delimiter=",")
#     file.close()
#     problem: TravelingSalesPersonProblem = TravelingSalesPersonProblem(distanceMatrix)
#     parent_one = TravelingSalesPersonIndividual()
#     parent_two = TravelingSalesPersonIndividual()
#     parent_one.set_order(try_list1)
#     parent_two.set_order(try_list2)
#
#     print("Running...")
#     offspring,inters = DPX(distanceMatrix,parent_one,parent_two)
#     oo = offspring.get_order()
#     for i in range(len(inters)):
#         ind = oo.index(inters[i][0])
#         print("value: %s and value %s " % (oo[ind],oo[ind+1]))
#     b = []
#     for i in oo:
#         a = []
#         for y in range(929):
#             if i == oo[y]:
#                 a.append(i)
#
#         if len(a) == 2:
#             b.append(a[0])
#     print("value occuring twice: %s" % b)
#
#     m = []
#     for x in range(929):
#         if x not in set(oo):
#             m.append(x)
#
#     print("city missing in oo: %s" % m)
#
#     print(oo)
#     print(len(oo))
#     print(len(set(oo)))
#
#
#
#
#     print("Finished")
#
#
#
# run()

