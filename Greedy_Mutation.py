import numpy as np
from representation import TravelingSalesPersonIndividual
from copy import deepcopy
from random import randint,choice,random

def Greedy_Mutation(cost_matrix_original, parent_one:TravelingSalesPersonIndividual, mutation_rate:float):
    if random() > mutation_rate:
        # return parent_one
        order1 = parent_one.get_order()
        cost1 = parent_one.get_cost()
        return order1,cost1


    else:

        order1 = parent_one.get_order()
        amount_of_cities_to_visit = len(cost_matrix_original)

        visited_edges1 = [(order1[i], order1[i + 1]) for i in range(0, len(order1) - 1)]
        visited_edges1 += [(order1[len(order1) - 1], order1[0])]

        # Number between 4 and 7.
        number = randint(4, 7)
        order = parent_one.get_order()
        visited_edges = [(order[i], order[i + 1]) for i in range(0, len(order) - 1)]
        visited_edges += [(order[len(order) - 1], order[0])]
        used_indices = []

        remove_edge_index = randint(0, amount_of_cities_to_visit - 1)
        for n in range(number):
            while remove_edge_index in used_indices:
                remove_edge_index = randint(0, amount_of_cities_to_visit - 1)
            used_indices.append(remove_edge_index)

        # print("used indices for mutation: %s" % used_indices)

        inters = [e for i, e in enumerate(visited_edges) if i not in used_indices]

        amount_of_cities_to_visit = len(cost_matrix_original)

        if len(inters) == amount_of_cities_to_visit:
            # offspring = TravelingSalesPersonIndividual()
            # offspring.set_order(order1)
            cost1 = parent_one.get_cost()
            # offspring.set_cost(cost1)
            # return offspring
            return order1,cost1
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
                # print(inters)
                # print(y[0])


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
        if len(set(NN_route)) != amount_of_cities_to_visit:
            raise Exception("error: in the amount of cities to visit")

        # return offspring,inters,NN_cost
        # return offspring
        return NN_route,NN_cost

def fun(current_city,list_edges):

    (sc,ec) =  list_edges
    if current_city == sc:
        return True

    else:
        return False


# def run(filename = "tour29.csv"):
#     import numpy as np
#     from random import shuffle
#     from representation import TravelingSalesPersonIndividual, TravelingSalesPersonProblem
#     file = open(filename)
#     distanceMatrix = np.loadtxt(file, delimiter=",")
#     file.close()
#     problem: TravelingSalesPersonProblem = TravelingSalesPersonProblem(distanceMatrix)
#     count = 1
#     for _ in range(3000):
#         try_list1 = list(range(29))
#
#         shuffle(try_list1)
#         print("the original list: %s" % try_list1)
#
#
#
#         parent_one = TravelingSalesPersonIndividual()
#         parent_one.set_order(try_list1)
#
#
#         print("Running...")
#         offspring,inters,NN_cost = Greedy_Mutation(distanceMatrix,parent_one)
#         oo = offspring.get_order()
#         print("inters: %s" % inters)
#         # for i in range(len(inters)):
#         #     ind = oo.index(inters[i][0])
#         #     # print("value: %s and value %s " % (oo[ind],oo[ind+1]))
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
#         print("after mutation: %s"% oo)
#         print(len(oo))
#         print(len(set(oo)))
#         if len(set(oo)) != 29:
#             raise Exception("error")
#         print("NN_cost: %s" % NN_cost)
#         print("Finished")
#         count += 1
#
#     print(count)
#
#
# run()
#
