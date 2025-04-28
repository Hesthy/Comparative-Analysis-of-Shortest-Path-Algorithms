import SPAlgorithm
import Graph

class ShortPathFinder:
    def __init__(self):
        self.graph = None
        self.alogrithm = None
    
    def calc_short_path(self,source,dest):
        result = self.algorithm.calc_sp(self.graph, source, dest)

        """
        Since the original algorithm returns a tuple of (distance, path) for Dijkstra and Bellmanford, and just the path for A*,
        for the sake of respect to the original code, we choose to not change the return value of the algorithm.
        Therefore, we handle the inconsistency of return values here.
        If the length is 2, we assume it's Dijkstra or Bellmanford, and we return the path by accessing the second element of the tuple.
        If the length is 1, we assume it's A*, and we return the path directly.
        """
        if result is None:
            return None
        if isinstance(result, tuple):
            return result[1][dest]
        return result

    def set_algorithm(self, algorithm:SPAlgorithm):
        self.algorithm = algorithm
    
    def set_graph(self,graph:Graph):
        self.graph = graph

def main(*args): # it is a directed graph
    g = Graph.WeightedGraph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    g.add_node(3)
    g.add_node(4)

    g.add_edge(0, 1, 1)
    g.add_edge(1, 2, 2)
    g.add_edge(2, 3, 1)
    g.add_edge(3, 4, 3)
    g.add_edge(0, 2, 4)

    print("Graph: ", g.adj) # should be {0: {1: {'weight': 1}, 2: {'weight': 4}}, 1: {2: {'weight': 2}}, 2: {3: {'weight': 1}}, 3: {4: {'weight': 3}}}

    h = Graph.HeuristicGraph()
    h.add_node(0)
    h.add_node(1)
    h.add_node(2)
    h.add_node(3)
    h.add_node(4)

    h.add_edge(0, 1, 1)
    h.add_edge(1, 2, 2)
    h.add_edge(2, 3, 1)
    h.add_edge(3, 4, 3)
    h.add_edge(0, 2, 4)

    h._heuristic = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0} # heuristic values for A*

    spf = ShortPathFinder()
    spf.set_graph(g)
    spf.set_algorithm(SPAlgorithm.Dijkstra())
    pathdij = spf.calc_short_path(0, 1)
    print("Shortest path from 0 to 1 using Dijkstra:", pathdij) # should be [0, 1]
    spf.set_algorithm(SPAlgorithm.BellmanFord())
    pathbell = spf.calc_short_path(0, 1)
    print("Shortest path from 0 to 1 using Bellman-Ford:", pathbell) # should be [0, 1]
    spf.set_algorithm(SPAlgorithm.AStar(h.get_heuristic))
    pathastar = spf.calc_short_path(0, 1)
    print("Shortest path from 0 to 1 using A*:", pathastar) # should be [0, 1]


main()