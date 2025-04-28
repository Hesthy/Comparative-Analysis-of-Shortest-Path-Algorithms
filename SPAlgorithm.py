import heapq
import Graph

class SPAlgorithm:
    def calc_sp(self, graph, source, dest):
        raise NotImplementedError("This method should be overridden by subclasses")

class Dijkstra(SPAlgorithm):
    def __init__(self, k=10000):
        self.k = k

    def calc_sp(self, graph:Graph, source, dest):
        k = self.k
        distance = {node:float('inf') for node in graph.adj}
        distance[source] = 0

        path = {node: [] for node in graph.adj}
        path[source] = [source]

        relax_count = {node: 0 for node in graph.adj} #counter
        
        priority_queue = [(0, source)]  # (distance, node)

        while priority_queue:
            dist_u, u = heapq.heappop(priority_queue) #pop the node with the smallest distance
            if relax_count[u] >= k: #if the node has been relaxed k times, skip
                continue    

            for v in graph.adj[u]:  
                weight = graph.get_adj_nodes(u)[v]['weight']
                if distance[v] > distance[u] + weight:
                    distance[v] = distance[u] + weight
                    path[v] = path[u] + [v]
                    heapq.heappush(priority_queue, (distance[v], v))
                    relax_count[v] += 1
        
        return distance, path

class AStar(SPAlgorithm):
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def calc_sp(self, graph, source, dest):
        heuristic = self.heuristic
        open_list = [(0, source)]  #to record the node 
        closed_list = set() #to check visited node

        g_values = {node: float('inf') for node in range(len(graph.adj))} #set every cost to infinity 
        g_values[source] = 0

        parent = {source: None}

        while open_list:
            current_f, current_node = heapq.heappop(open_list) # variable to receive the smallest value in the list
        
            if current_node == dest:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = parent[current_node]
                return path[::-1] 

            closed_list.add(current_node)


            for neighbor_raw in graph.adj[current_node]:
                neighbor = int(neighbor_raw)  

                if neighbor in closed_list:
                    continue

                if neighbor not in g_values:
                    g_values[neighbor] = float('inf')
            
                cost = graph.adj[current_node][neighbor]['weight']
                tentative_g = g_values[current_node] + cost

                if tentative_g < g_values[neighbor]:
                    g_values[neighbor] = tentative_g
                    f_value = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_list, (f_value, neighbor))
                    parent[neighbor] = current_node

        return None  #if no path found
    
class BellmanFord(SPAlgorithm):
    def __init__(self, k=10000):
        self.k = k

    def calc_sp(self, graph, source, dest):
        k = self.k
        distance = {node: float('inf') for node in range(len(graph.adj))}
        distance[source] = 0

        path = {node: [] for node in range(len(graph.adj))}
        path[source] = [source]

        for _ in range(k):  # at most k iterations
            update = False
            for u in graph.adj:
                for v in graph.adj[u]:
                    weight = graph.get_adj_nodes(u)[v]['weight']
                    if distance[u] + weight < distance[v]:
                        distance[v] = distance[u] + weight
                        path[v] = path[u] + [v]
                        update = True
            if not update:
                break

        return distance, path