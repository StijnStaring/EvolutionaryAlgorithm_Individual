
class TravelingSalesPersonIndividual:

    def __init__(self):
        self.order = []
        self.visited_vertices = set()

    def get_order(self):
        return self.order

    def get_visited_vertices(self):
        return self.visited_vertices

    def set_order(self, new_order):
        self.order = new_order
        self.visited_vertices = set(new_order)

    def __str__(self) -> str:
        return str(self.get_order())

class TravelingSalesPersonProblem:

    def __init__(self, weights):
        self.weights = weights
        self.visited_vertices = {i for i in range(len(weights))}

    def get_weight(self, a, b):
        return self.weights[a][b]

    def get_dimension(self) -> int:
        return len(self.weights)

    def calculate_individual_score(self, individual: TravelingSalesPersonIndividual):
        # Check if the individual visits all nodes.
        if self.visited_vertices != individual.get_visited_vertices():
            return float('inf')
        
        order = individual.get_order()
        if len(order) == 0:
            # No path at all ==> no weights
            return 0
        visited_edges = [(order[i], order[i + 1]) for i in range(0, len(order) - 1)]
        visited_edges += [(order[len(order) - 1], order[0])]

        path_weight = 0
        for (a,b) in visited_edges:
            path_weight += self.get_weight(a,b)

        return path_weight



