import numpy as np
from representation import TravelingSalesPersonIndividual
from copy import deepcopy
from random import randint,choice

def DPX(cost_matrix_original, parent_one:TravelingSalesPersonIndividual, parent_two:TravelingSalesPersonIndividual):
    order1 = parent_one.get_order()
    order2 = parent_two.get_order()

    visited_edges1 = [(order1[i], order1[i + 1]) for i in range(0, len(order1) - 1)]
    visited_edges1 += [(order1[len(order1) - 1], order1[0])]

    visited_edges2 = [(order2[i], order2[i + 1]) for i in range(0, len(order2) - 1)]
    visited_edges2 += [(order2[len(order2) - 1], order2[0])]

#     Look for the edges in common.
    inters = list(set(visited_edges1).intersection(set(visited_edges2)))
    amount_of_cities_to_visit = len(cost_matrix_original)

    if len(inters) == amount_of_cities_to_visit:
        offspring = TravelingSalesPersonIndividual()
        offspring.set_order(order1)
        cost1 = parent_one.get_cost()
        offspring.set_cost(cost1)
        return offspring
#     inters = [(918, 323), (685, 606),(700,701)]
#     print("inters is: %s " % inters)

    available_choices = [i for i in range(amount_of_cities_to_visit)]
    for (i,j) in inters:
        available_choices.remove(j)
    random_city = choice(available_choices)

#     Build the offspring
    cost_matrix = deepcopy(cost_matrix_original)

    NN_route = [random_city]

    # NN_edges = []
    # check: bool = False
    # random_city = 0
    # while not check:
    #     print("random city: %s" % random_city)
    #     print(inters)
    #     random_city = randint(0, amount_of_cities_to_visit - 1)
    #     if all(list(map(lambda x: x[1] != random_city,inters))):
    #         check = True

    current_city = random_city
    NN_cost = 0
    y = []
    while len(NN_route) != amount_of_cities_to_visit:
        if len(inters) != 0:
            y = list(filter(lambda x: fun(current_city,x),inters))


        costs = cost_matrix[current_city]

        if len(y) != 0:
            next_city = y[0][1]
            NN_cost += costs[next_city]
            # NN_edges.append((current_city,next_city))
            NN_route.append(next_city)
            current_city = next_city


        else:
            if len(inters) != 0:
                for (c1, c2) in inters:
                    costs[c2] = np.inf

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

    # return offspring,inters,NN_cost
    return offspring

def fun(current_city,list_edges):

    (sc,ec) =  list_edges
    if current_city == sc:
        return True

    else:
        return False

#
# def run(filename = "tour29.csv"):
#     import numpy as np
#     from random import shuffle
#     from representation import TravelingSalesPersonIndividual, TravelingSalesPersonProblem
#     file = open(filename)
#     distanceMatrix = np.loadtxt(file, delimiter=",")
#     file.close()
#     problem: TravelingSalesPersonProblem = TravelingSalesPersonProblem(distanceMatrix)
#
#     for i in range(300):
#         try_list1 = list(range(29))
#         try_list2 = list(range(29))
#         shuffle(try_list1)
#         shuffle(try_list2)
#
#
#         parent_one = TravelingSalesPersonIndividual()
#         parent_two = TravelingSalesPersonIndividual()
#         parent_one.set_order(try_list1)
#         parent_two.set_order(try_list2)
#
#         print("Running...")
#         offspring,inters,NN_cost = DPX(distanceMatrix,parent_one,parent_two)
#         oo = offspring.get_order()
#         print("inters: %s" % inters)
#         for i in range(len(inters)):
#             ind = oo.index(inters[i][0])
#             print("value: %s and value %s " % (oo[ind],oo[ind+1]))
#         b = []
#         for i in oo:
#             a = []
#             for y in range(29):
#                 if i == oo[y]:
#                     a.append(i)
#
#             if len(a) == 2:
#                 b.append(a[0])
#         print("value occuring twice: %s" % b)
#
#         m = []
#         for x in range(29):
#             if x not in set(oo):
#                 m.append(x)
#
#         print("city missing in oo: %s" % m)
#
#         print(oo)
#         print(len(oo))
#         print(len(set(oo)))
#         if len(set(oo)) != 29:
#             raise Exception("error")
#         print("NN_cost: %s" % NN_cost)
#         print("Finished")
#
#
#
# run()

