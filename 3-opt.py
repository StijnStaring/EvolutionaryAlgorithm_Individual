import itertools
from representation import TravelingSalesPersonProblem
from random import shuffle,randint
from copy import deepcopy

def opt_3_local_search(problem: TravelingSalesPersonProblem, route:list, max_iterations:int = 1000):
    """
    3 OPT Local search
    route: list of cities
    max_swaps: max amount of swaps
    updated list of cities, tour_cost
    # local search is zo geschreven dat het bijna onafhankelijk is van de lengte van de tour
    # Vooral enkel in het begin hebt verbetering
    """
    amount_of_cities = len(route)
    iteration = 1
    tried_combinations = []
    distance: float = 0
    swap: bool = False

    # Don't check possibilities --> assume that not possible to check them all
    # possibilities = 0
    # for i in range(amount_of_cities):
    #     for j in range(i + 1, amount_of_cities):
    #         for k in range(j + 1, amount_of_cities):
    #             possibilities += 1

    while iteration <= max_iterations:

        random_edge1 = 0
        random_edge2 = 0
        random_edge3 = 0
        check: bool = False
        it = 0
        while (random_edge1 == random_edge2 and random_edge1 == random_edge3 and random_edge2 == random_edge3) or (not check):

            random_edge1 = randint(0,amount_of_cities - 1)
            random_edge2 = randint(0, amount_of_cities - 1)
            random_edge3 = randint(0, amount_of_cities - 1)

            if all(list(map(lambda x: x not in tried_combinations,itertools.permutations([random_edge1,random_edge2,random_edge3], 3)))):
                check = True

            if it == 1000:
                raise Exception("Can't find a new candidate.")

            it += 1

        item = (random_edge1,random_edge2,random_edge3)
        tried_combinations.append(item)
        # all_combinations = list(itertools.combinations(range(len(route)), 3)) # all possible selections of edges
        # shuffle(all_combinations) # needed?

        # for item in all_combinations:
            # item contains three indices: index1,index2,index3

        route,distance,swap,improvement = generate_combinations(problem, route, item)
        print("The remaining distance: %s\tAmount of improvement: %s." % (distance,improvement))

        if swap:
            tried_combinations = []


        # if not swap: Assume that the algorithm will never reach all the possible swaps
        #     break

        iteration += 1

    return route, distance

def generate_combinations(problem: TravelingSalesPersonProblem,route,item):

    es1,es2,es3 = item
    ee1,ee2,ee3 = es1 +1, es2 + 1, es3 + 1
    length = len(route)
    indices_start = [es1,es2,es3]

    if ee1 >= length:
        ee1 = 0

    if ee2 >= length:
        ee2 = 0

    if ee3 >= length:
        ee3 = 0

    # Order that is shown here is of importance!
    combo1 = swapPositions(route,es1,ee1) #A'BC
    combo2 = swapPositions(route, es2, ee2)  # AB'C
    combo3 = swapPositions(route, es3, ee3)  # ABC'
    combo4_temp = swapPositions(route, es2, ee2)  # AB'C'
    combo4 = swapPositions(combo4_temp, es3, ee3)
    combo5_temp = swapPositions(route, es1, ee1)  # A'BC'
    combo5 = swapPositions(combo5_temp, es3, ee3)
    combo6_temp = swapPositions(route, es1, ee1)  # A'B'C
    combo6 = swapPositions(combo6_temp, es2, ee2)
    combo7_temp1 = swapPositions(route, es1, ee1)  # A'B'C'
    combo7_temp2 = swapPositions(combo7_temp1, es2, ee2)
    combo7 = swapPositions(combo7_temp2, es3, ee3)

    initial_cost = cost(problem, route)
    min_cand = route
    improvement = 0
    swap = True
    kind = 1
    for cand in [combo1, combo2, combo3, combo4, combo5, combo6, combo7]:

        min_cand,improvement = cost_effect(problem,route,cand,indices_start,kind,initial_cost,min_cand,improvement)
        kind += 1

    if improvement == 0:
        swap = False

    min_cost = initial_cost - improvement

    return min_cand,min_cost,swap,improvement

def cost_effect(problem,route,cand,indices_start,kind,initial_cost,min_cand,improvement):

    amount_cities = len(route)
    original_edges = set()
    new_edges = set()

    if kind == 1:
        Ias = indices_start[0]

        for value in [-1,0,1]:
            I1 = Ias + value
            I2 = Ias + 1 + value

            if I1 >= amount_cities:
                I1 -= amount_cities

            if I2 >= amount_cities:
                I2 -= amount_cities

            original_edges.add((route[I1],route[I2]))
            new_edges.add((cand[I1],cand[I2]))

    elif kind == 2:
        Ibs = indices_start[1]

        for value in [-1, 0, 1]:
            I1 = Ibs + value
            I2 = Ibs + 1 + value

            if I1 >= amount_cities:
                I1 -= amount_cities

            if I2 >= amount_cities:
                I2 -= amount_cities

            original_edges.add((route[I1], route[I2]))
            new_edges.add((cand[I1], cand[I2]))

    elif kind == 3:
        Ics = indices_start[2]

        for value in [-1, 0, 1]:
            I1 = Ics + value
            I2 = Ics + 1 + value

            if I1 >= amount_cities:
                I1 -= amount_cities

            if I2 >= amount_cities:
                I2 -= amount_cities

            original_edges.add((route[I1], route[I2]))
            new_edges.add((cand[I1], cand[I2]))

    elif kind == 4:

        for I in [indices_start[1], indices_start[2]]:
            for value in [-1, 0, 1]:
                I1 = I + value
                I2 = I + 1 + value

                if I1 >= amount_cities:
                    I1 -= amount_cities

                if I2 >= amount_cities:
                    I2 -= amount_cities

                original_edges.add((route[I1], route[I2]))
                new_edges.add((cand[I1], cand[I2]))

    elif kind == 5:
        for I in [indices_start[0], indices_start[2]]:
            for value in [-1, 0, 1]:
                I1 = I + value
                I2 = I + 1 + value

                if I1 >= amount_cities:
                    I1 -= amount_cities

                if I2 >= amount_cities:
                    I2 -= amount_cities

                original_edges.add((route[I1], route[I2]))
                new_edges.add((cand[I1], cand[I2]))

    elif kind == 6:
        for I in [indices_start[0], indices_start[1]]:
            for value in [-1, 0, 1]:
                I1 = I + value
                I2 = I + 1 + value

                if I1 >= amount_cities:
                    I1 -= amount_cities

                if I2 >= amount_cities:
                    I2 -= amount_cities

                original_edges.add((route[I1], route[I2]))
                new_edges.add((cand[I1], cand[I2]))

    elif kind == 7:
        for I in [indices_start[0], indices_start[1], indices_start[2]]:
            for value in [-1, 0, 1]:
                I1 = I + value
                I2 = I + 1 + value

                if I1 >= amount_cities:
                    I1 -= amount_cities

                if I2 >= amount_cities:
                    I2 -= amount_cities

                original_edges.add((route[I1], route[I2]))
                new_edges.add((cand[I1], cand[I2]))



    original_path_weight = 0
    for (a, b) in original_edges:
        original_path_weight += problem.get_weight(a, b)

    new_path_weight = 0
    for (a, b) in new_edges:
        new_path_weight += problem.get_weight(a, b)

    new_improvement = original_path_weight - new_path_weight

    if new_improvement > improvement:
        return cand, new_improvement

    else:
        return min_cand,improvement


def cost(problem: TravelingSalesPersonProblem,order:list):
    visited_edges = [(order[i], order[i + 1]) for i in range(0, len(order) - 1)]
    visited_edges += [(order[len(order) - 1], order[0])]

    path_weight = 0
    for (a,b) in visited_edges:
        path_weight += problem.get_weight(a,b)

    return path_weight


def swapPositions(list_input:list, pos1, pos2):
    copy_list =  deepcopy(list_input)
    copy_list[pos1], copy_list[pos2] = copy_list[pos2], copy_list[pos1] # changes original list because do an operation on the elements
    return copy_list



def run(filename = "tour929.csv"):
    import numpy as np
    try_list = list(range(929))
    shuffle(try_list)
    file = open(filename)
    distanceMatrix = np.loadtxt(file, delimiter=",")
    file.close()
    problem: TravelingSalesPersonProblem = TravelingSalesPersonProblem(distanceMatrix)
    print("Running...")
    result,cost_out = opt_3_local_search(problem,try_list,10)

    print("Finished")
    return result


run()