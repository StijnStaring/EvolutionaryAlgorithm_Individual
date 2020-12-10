import numpy as np
from copy import deepcopy
from representation import TravelingSalesPersonProblem

def Opt_3(route_original:list,distanceMatrix,amount_cities,max_iterations=1):
    iteration = 0
    route = deepcopy(route_original)
    count = 0
    while count <= amount_cities and iteration < max_iterations:

        for i in range(amount_cities):

            """ Initialization of the current/next/previous cities """
            current_city = route[i]
            next_city:int
            next_city_i:int
            previous_city:int
            previous_city_i:int

            if i == 0:
                next_city = route[i + 1]
                next_city_i = i+1
                previous_city = route[amount_cities -1]
                previous_city_i = amount_cities - 1

            elif i == amount_cities -1:
                next_city = route[0]
                next_city_i = 0
                previous_city = route[i-1]
                previous_city_i = i - 1

            else:
                next_city = route[i+1]
                next_city_i = i + 1
                previous_city = route[i-1]
                previous_city_i = i - 1

            """ 3-opt forward case """
            costs1:list = distanceMatrix[current_city]
            current1:float = costs1[next_city]
            improvement1 = 0
            memory_cities1 = []

            candidates = [(c,co) for (c,co) in list(enumerate(costs1)) if co < current1 and c != previous_city and c != current_city]
            # print(candidates)
            sorted_candidates_start = sorted(candidates, key=lambda x: x[1])

            for (c,co) in sorted_candidates_start:
                c_index = route.index(c)
                if c_index == amount_cities-1:
                    city_D_i = 0
                else:
                    city_D_i = c_index+1
                if c_index == 0:
                    city_F_i = amount_cities -1
                else:
                    city_F_i = c_index -1
                city_D = route[city_D_i]
                city_F = route[city_F_i]
                costs_c = deepcopy(distanceMatrix[c])
                costs1_relative = current1 + costs_c[city_D] - costs1[c]

                if previous_city_i >= c_index:
                    not_desired_cities = [city_F,current_city, previous_city, next_city] + route[c_index:previous_city_i]
                else:
                    not_desired_cities = [city_F,current_city, previous_city, next_city] + route[c_index:] + route[:previous_city_i]

                costs_c[not_desired_cities] = np.inf
                candidates2 = [(c2, co) for (c2, co) in list(enumerate(costs_c)) if co < costs1_relative]
                sorted_candidates_end = sorted(candidates2, key=lambda x: x[1])

                for (city_end,cost_end) in sorted_candidates_end:
                    city_end_i = route.index(city_end)
                    if city_end_i == len(route) - 1:
                        city_end_next_i = 0
                    else:
                        city_end_next_i = city_end_i + 1

                    city_end_next = route[city_end_next_i]

                    cost_original = 0
                    cost_reversed = 0

                    insert_part: list
                    if city_D_i >= city_end_next_i:
                        insert_part = route[city_end_next_i:city_D_i]
                    else:
                        insert_part = route[city_end_next_i:] + route[:city_D_i]

                    visited_edges = [(insert_part[i], insert_part[i + 1]) for i in range(0, len(insert_part) - 1)]
                    for (a, b) in visited_edges:
                        cost_original += distanceMatrix[a, b]

                    insert_part.reverse()
                    visited_edges = [(insert_part[i], insert_part[i + 1]) for i in range(0, len(insert_part) - 1)]
                    for (a, b) in visited_edges:
                        cost_reversed += distanceMatrix[a, b]

                    improvement1 = costs1_relative + distanceMatrix[city_end][city_end_next] - distanceMatrix[city_end_next][next_city]  - distanceMatrix[city_end][city_D] + cost_original - cost_reversed
                    memory_cities1 = [city_end_next_i, city_D_i,c,city_end]
                    if improvement1 > 0:
                        break
                if improvement1 > 0:
                    break


            """ 3-opt backward case """

            costs1 = distanceMatrix[:,current_city]
            current1 = costs1[previous_city]
            improvement2 = 0
            memory_cities2 = []

            candidates = [(c, co) for (c, co) in list(enumerate(costs1)) if co < current1 and c != next_city and c != current_city]
            sorted_candidates_start = sorted(candidates, key=lambda x: x[1])

            for (c, co) in sorted_candidates_start:
                c_index = route.index(c)

                if c_index == amount_cities - 1:
                    city_D_i = 0
                else:
                    city_D_i = c_index + 1

                if c_index == 0:
                    city_F_i = amount_cities - 1
                else:
                    city_F_i = c_index - 1

                city_D = route[city_D_i]
                city_F = route[city_F_i]

                costs_c = deepcopy(distanceMatrix[:,c]) # cost to travel to c
                costs1_relative = current1 + distanceMatrix[c,city_D] - distanceMatrix[c][current_city]

                if previous_city_i >= c_index:
                    not_desired_cities = [city_F, current_city, previous_city, next_city] + route[c_index:previous_city_i]
                else:
                    not_desired_cities = [city_F,current_city, previous_city, next_city] + route[c_index:] + route[:previous_city_i]

                costs_c[not_desired_cities] = np.inf
                candidates2 = [(c2, co) for (c2, co) in list(enumerate(costs_c)) if co < costs1_relative]
                sorted_candidates_end = sorted(candidates2, key=lambda x: x[1])

                for (city_end, cost_end) in sorted_candidates_end:
                    city_end_i = route.index(city_end)
                    if city_end_i == len(route) - 1:
                        city_end_next_i = 0
                    else:
                        city_end_next_i = city_end_i + 1

                    city_end_next = route[city_end_next_i]

                    improvement2 = costs1_relative + distanceMatrix[city_end][city_end_next] - distanceMatrix[city_end][city_D] - distanceMatrix[previous_city][city_end_next]
                    memory_cities2 = [city_end_next_i, city_D_i,c,city_end]

                    if improvement2 > 0:
                        break
                if improvement2 > 0:
                    break

            """ Swapping best suggestion """
            if improvement1 <= 0 and improvement2 <= 0:
                count += 1
            else:
                count = 0

            if improvement1 >= improvement2 and improvement1 > 0:

                city_end_next_i = memory_cities1[0]
                city_D_i = memory_cities1[1]

                insert_part:list
                if city_D_i >= city_end_next_i:
                    insert_part = route[city_end_next_i:city_D_i]
                    insert_part.reverse()

                else:
                    insert_part = route[city_end_next_i:] + route[:city_D_i]
                    insert_part.reverse()

                if city_end_next_i>=next_city_i:
                    part1 = route[next_city_i:city_end_next_i]
                else:
                    part1 = route[next_city_i:] + route[:city_end_next_i]

                if previous_city_i >= city_D_i:
                    part2 = route[city_D_i:previous_city_i+1]
                else:
                    part2 = route[city_D_i:] + route[:previous_city_i+1]

                route = [current_city] + insert_part + part1 + part2

            elif improvement1 < improvement2 and improvement2 > 0:
                city_end_next_i = memory_cities2[0]
                city_D_i = memory_cities2[1]

                insert_part: list
                if city_D_i >= city_end_next_i:
                    insert_part = route[city_end_next_i:city_D_i]

                else:
                    insert_part = route[city_end_next_i:] + route[:city_D_i]

                if city_end_next_i >= next_city_i:
                    part1 = route[next_city_i:city_end_next_i]
                else:
                    part1 = route[next_city_i:] + route[:city_end_next_i]

                if previous_city_i >= city_D_i:
                    part2 = route[city_D_i:previous_city_i+1]
                else:
                    part2 = route[city_D_i:] + route[:previous_city_i+1]

                route = [current_city] + part1 + part2 + insert_part

        iteration += 1


    return route
