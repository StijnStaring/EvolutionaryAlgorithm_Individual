from representation import TravelingSalesPersonIndividual,TravelingSalesPersonProblem
from random import randint,choice,random
import numpy as np

def Greedy_Mutation_2(distanceMatrix, parent_one:TravelingSalesPersonIndividual, mutation_rate:float, amount_of_cities_to_visit:int):

    if random() > mutation_rate:
        # return parent_one
        order = parent_one.get_order()
        cost = parent_one.get_cost()
        return order,cost

    else:

        # Number of edges that are going to be removed is between 4 and 7.
        number = randint(4, 7)
        order = parent_one.get_order()
        cost1 = parent_one.get_cost()
        # print("cost original: %s" % cost1)
        used_indices = []
        available_choice = list(range(amount_of_cities_to_visit))

        for _ in range(number):
            remove_edge_index = choice(available_choice)
            used_indices.append(remove_edge_index)
            available_choice.remove(remove_edge_index)

        used_indices = sorted(used_indices)

        used_indices = [1,3,27,28]

        old_cost = 0
        next_city:int
        for ui in used_indices:
            if ui != amount_of_cities_to_visit-1:
                next_city = order[ui + 1]
            else:
                next_city = order[0]
            old_cost += distanceMatrix[order[ui]][next_city]

        parts = []
        for i in range(len(used_indices) - 1):
            parts.append(order[used_indices[i]+1:used_indices[i+1]+1])
        part1 = order[used_indices[-1]+1:] + order[:used_indices[0]+1]
        parts.insert(0,part1)

        build = parts[0]
        new_cost = 0
        rest = parts[1:]
        for i in range(len(used_indices) - 1):
            endpoints = list(build[-1]*np.ones(len(rest)))
            startpoints_rest = [s[0] for s in rest]
            combi = list(map(lambda x,y: (x,y),endpoints,startpoints_rest))
            costs_rest = [get_cost_mutation(int(a),int(b),distanceMatrix) for (a,b) in combi]
            chosen_part_index = np.argmin(costs_rest)
            new_cost = new_cost +  costs_rest[int(chosen_part_index)]
            build = build + rest[int(chosen_part_index)]
            del rest[int(chosen_part_index)]

        new_cost += distanceMatrix[build[-1]][build[0]]

        # problem: TravelingSalesPersonProblem = TravelingSalesPersonProblem(distanceMatrix)
        # KOST = cost10(problem,build)
        # print("Difference: %s" % (cost1 + new_cost - old_cost-KOST))

        return build,(cost1 + new_cost - old_cost)



def get_cost_mutation(a,b,distanceMatrix):
    return distanceMatrix[a][b]

# def cost10(problem: TravelingSalesPersonProblem,order:list):
#     visited_edges = [(order[i], order[i + 1]) for i in range(0, len(order) - 1)]
#     visited_edges += [(order[len(order) - 1], order[0])]
#
#     path_weight = 0
#     for (a,b) in visited_edges:
#         path_weight += problem.get_weight(a,b)
#
#     return path_weight

# def run2(filename = "tour29.csv"):
#     import numpy as np
#     from random import shuffle
#     from representation import TravelingSalesPersonIndividual, TravelingSalesPersonProblem
#     file = open(filename)
#     distanceMatrix = np.loadtxt(file, delimiter=",")
#     file.close()
#     problem: TravelingSalesPersonProblem = TravelingSalesPersonProblem(distanceMatrix)
#     for _ in range(3000):
#         try_list1 = list(range(29))
#
#         shuffle(try_list1)
#         print("the original list: %s" % try_list1)
#
#         parent_one = TravelingSalesPersonIndividual()
#         parent_one.set_order(try_list1)
#         cp1 = problem.calculate_individual_score(parent_one)
#         parent_one.set_cost(cp1)
#
#
#         print("Running...")
#         offspring_order,offspring_cost = Greedy_Mutation_2(distanceMatrix,parent_one,1.0,len(distanceMatrix))
#         offspring = TravelingSalesPersonIndividual()
#         offspring.set_cost(offspring_cost)
#         offspring.set_order(offspring_order)
#         oo = offspring.get_order()
#
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
#         print("NN_cost: %s" % offspring_cost)
#         cost_check = cost10(problem,offspring_order)
#         print("kost_check: %s" % cost_check)
#         print("Finished")
#
#
# run2()
