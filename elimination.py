def elimination(problem, list_of_individuals: list, amount_of_individuals_to_choose: int) -> list:
    list_of_individuals = sorted(list_of_individuals, key=lambda individual: problem.calculate_individual_score(individual))
    return list_of_individuals[:amount_of_individuals_to_choose]
