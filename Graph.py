class Graph:
    def __init__(self):
        self.adj = {}

    def get_adj_nodes(self, node):
        return self.adj[node]
    
    def add_node(self, node):
        self.adj[node] = {}

    def add_edge(self, start, end, w):
        if start not in self.adj or end not in self.adj:
            raise ValueError("Node not found in graph")

        self.adj[start][end] = {"weight": w}  

    def get_num_of_nodes(self):
        return len(self.adj)

    def w(self, node):
        weights = {}

        if node not in self.adj:
            return weights
        
        for n, w in self.adj[node]:
            weights[n] = w

        return weights
    
class WeightedGraph(Graph):
    def w(self, node1, node2):
        if node1 not in self.adj or node2 not in self.adj:
            return None

        if node2 not in self.adj[node1]:
            return None
        
        return self.adj[node1][node2]["weight"]
    
class HeuristicGraph(WeightedGraph):
    def __init__(self):
        super().__init__()
        self._heuristic = None
    
    def get_heuristic(self, int):
        return self._heuristic[int]