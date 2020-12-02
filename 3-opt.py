import itertools
from representation import TravelingSalesPersonProblem
from random import shuffle
from copy import deepcopy

def opt_3_local_search(problem: TravelingSalesPersonProblem, route:list, max_swaps:int = 200):
    """
    3 OPT Local search
    route: list of cities
    max_swaps: max amount of swaps
    updated list of cities, tour_cost
    """
    iteration = 1
    distance: float = 0
    swap: bool = False
    while iteration <= max_swaps:
        all_combinations = list(itertools.combinations(range(len(route)), 3)) # all possible selections of edges
        shuffle(all_combinations) # needed?

        for item in all_combinations:
            # item contains three indices: index1,index2,index3

            route,distance,swap = generate_combinations(problem, route, item)

            if swap:
                break

        if not swap:
            break

        iteration += 1

    return route, distance

def generate_combinations(problem: TravelingSalesPersonProblem,route,item):

    es1,es2,es3 = item
    ee1,ee2,ee3 = es1 +1, es2 + 1, es3 + 1
    length = len(route)
    indices_start = [es1,ee1,es2]

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

    return min_cand,min_cost,swap

def cost_effect(problem,route,cand,indices_start,kind,initial_cost,min_cand,improvement):

    amount_cities = len(route)
    original_edges = set()
    new_edges = set()

    if kind == 1:
        Ias = indices_start[0]

        for value in [-1,2,1]:
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

        for value in [-1, 2, 1]:
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

        for value in [-1, 2, 1]:
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
            for value in [-1, 2, 1]:
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
            for value in [-1, 2, 1]:
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
            for value in [-1, 2, 1]:
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
            for value in [-1, 2, 1]:
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


def cost(problem: TravelingSalesPersonProblem,order):
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