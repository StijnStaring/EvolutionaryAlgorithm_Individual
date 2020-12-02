import numpy as np
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

            route, distance,swap = generate_combinations(problem, route, item)

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
    indices = [es1,ee1,es2,ee2,es3,ee3]

    if ee1 >= length:
        ee1 = 0

    if ee2 >= length:
        ee2 = 0

    if ee3 >= length:
        ee3 = 0

    if len({es1,es2,es3,ee1,ee2,ee3}) != 6:
        raise Exception("Should be 6 different indices!")

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

    min_cand = route
    min_cost = cost(problem,route)
    cost_route = min_cost
    swap = True
    kind = 1
    for cand in [combo1, combo2, combo3, combo4, combo5, combo6, combo7]:
        improvement = False
        min_cand,min_cost,improvement = cost_effect(cost_route,min_cost,cand,indices,kind)

        if improvement:
            min_cand = cand

        kind += 1

    if min_cost == cost_route:
        swap = False

    return min_cand,min_cost,swap

def cost_effect(cost_route,min_cost,cand,indices,kind):

    if kind <= 3: # 1 swap

    elif kind <= 6: # 2 swaps

    else: # 3 swaps








# can also go over everything and calculate the minimum --> more exploitation
def cost(problem: TravelingSalesPersonProblem,order): # improve the cost calculation --> not go over everything but only check what is changed in comparence with the previous calculation
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