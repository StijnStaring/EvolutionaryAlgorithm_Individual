import random as r
from representation import TravelingSalesPersonIndividual
import time

def scrambleMutate(individual: TravelingSalesPersonIndividual, alpha: float) -> TravelingSalesPersonIndividual:
    if individual.get_order()[0] != 0:
        print("Still a problem!")
        time.sleep(1)
        exit(0)
    
    singleRoute = individual.get_order()
    routeSubset = singleRoute[1:] # First zero removed
    shuffleEndPoints= r.sample(list(range(len(routeSubset))), 2)
    startPoint = min(shuffleEndPoints)
    finishPoint = max(shuffleEndPoints)
    randf = r.random()
    if startPoint == 0 and randf <= alpha: # In principle can also directly give singleRoute here
        subset1 = routeSubset[startPoint:finishPoint + 1]
        subset2 = routeSubset[finishPoint + 1:]
        subset1.extend(subset2)
        subset1.insert(0, 0)
        individual = TravelingSalesPersonIndividual()
        individual.set_order(subset1)
        return individual
    elif startPoint != 0 and randf <= alpha:
        subset1 = routeSubset[:startPoint]
        shuffleSubset = routeSubset[startPoint:finishPoint + 1]
        r.shuffle(shuffleSubset)
        subset2 = routeSubset[finishPoint + 1:]
        shuffleSubset.extend(subset2)
        subset1.extend(shuffleSubset)
        subset1.insert(0, 0)
        individual = TravelingSalesPersonIndividual()
        individual.set_order(subset1)
        return individual
    else:
        individual = TravelingSalesPersonIndividual()
        individual.set_order(singleRoute)
        return individual

def inverseMutate(individual, alpha):
    if individual.get_order()[0] != 0:
        print("Still a problem!")
        time.sleep(1)
        exit(0)
    
    singleRoute = individual.get_order()
    routeSubset = singleRoute[1:-1] # Remove first (zero) and last element (unknown)
    shuffleEndPoints= r.sample(list(range(len(routeSubset))), 2)
    startPoint = min(shuffleEndPoints)
    finishPoint = max(shuffleEndPoints)
    randf = r.random()
    if startPoint == 0 and randf <= alpha:
        subset1 = routeSubset[startPoint:finishPoint + 1]
        subset1.reverse()
        subset2 = routeSubset[finishPoint + 1:]
        subset1.extend(subset2)
        subset1.insert(0, 0)
        subset1.insert(len(subset1), 0) # Zero twice inserted and removed element not added
        individual = TravelingSalesPersonIndividual()
        individual.set_order(subset1)
        return individual
    elif startPoint != 0 and randf <= alpha:
        subset1 = routeSubset[:startPoint]
        middleSubset = routeSubset[startPoint:finishPoint + 1]
        middleSubset.reverse()
        subset2 = routeSubset[finishPoint + 1:]
        middleSubset.extend(subset2)
        subset1.extend(middleSubset)
        subset1.insert(0, 0)
        subset1.insert(len(subset1), 0) # Zero twice inserted and removed element not added
        individual = TravelingSalesPersonIndividual()
        individual.set_order(subset1)
        return individual
    else:
        individual = TravelingSalesPersonIndividual()
        individual.set_order(singleRoute)
        return individual


