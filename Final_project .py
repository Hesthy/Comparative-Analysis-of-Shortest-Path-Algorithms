import random
import time
import timeit 
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import pandas as pd
import math

#part2.1
def dijkstra (graph, source, k):
    distance = {node:float('inf') for node in graph}
    distance[source] = 0

    path = {node: [] for node in graph}
    path[source] = [source]

    relax_count = {node: 0 for node in graph} #counter
    
    priority_queue = [(0, source)]  # (distance, node)

    while priority_queue:
        dist_u, u = heapq.heappop(priority_queue) #pop the node with the smallest distance
        if relax_count[u] >= k: #if the node has been relaxed k times, skip
            continue    

        for v in graph[u]:  
            weight = graph[u][v]['weight']  
            if distance[v] > distance[u] + weight:
                distance[v] = distance[u] + weight
                path[v] = path[u] + [v]
                heapq.heappush(priority_queue, (distance[v], v))
                relax_count[v] += 1
    
    return distance, path

#part2.2
def bellman_ford(graph, source, k):
    distance = {node: float('inf') for node in range(len(graph))}
    distance[source] = 0

    path = {node: [] for node in range(len(graph))}
    path[source] = [source]

    for _ in range(k):  # at most k iterations
        update = False
        for u in graph:
            for v in graph[u]:
                weight = graph[u][v]['weight']
                if distance[u] + weight < distance[v]:
                    distance[v] = distance[u] + weight
                    path[v] = path[u] + [v]
                    update = True
        if not update:
            break

    return distance, path


# def create_random_graph(nodes, edges):
#     graph = {} # {Parent : [Children]}
#     for i in range(nodes):
#         graph[i] = [] #Initialize an empty neighbor list for each node
#     edge_count = 0
#     while edge_count < edges: 
#         parent = random.randint(0, nodes-1) #randomly select a parent node
#         child = random.randint(0, nodes-1)
#         if child not in graph[parent] and parent != child:
#             graph[parent].append(child) #add edge into graph
#             graph[child].append(parent)
#             edge_count += 1
#     return graph

# def weighted_graph(graph, weight_max): # function to generate random weights for the edges, should be in {src : [(dst, weight)]}
#     for i in graph:
#         for j in range(len(graph[i])):
#             graph[i][j] = (graph[i][j], random.randint(1, weight_max))
#     return graph

#part3
def allpair_dijkstra(G): # dijkstra complexity is O(E + VlogV) = O(E) for sparse graph, O(V^2) for dense graph, therefore expected complexity of allpair is O(V*(E + VlogV)) = O(VE) for sparse graph, O(V^3) for dense graph
    # O(V^3) for dense graph, O(VE) for sparse graph
    result = [[-1 for _ in range(len(G))] for _ in range(len(G))]
    for src in G:
        distance, _ = dijkstra(G, src, len(G))
        for dst in G:
            if src != dst:
                result[src][dst] = distance[dst]
    return result

def allpair_bellman_ford(G): # bellman-ford complexity is O(VE) for sparse graph, O(V^2) for dense graph, therefore expected complexity of allpair is O(V*(VE)) = O(V^2E) for sparse graph, O(V^3) for dense graph
    # O(V^2E) for sparse graph, O(V^3) for dense graph
    result = [[-1 for _ in range(len(G))] for _ in range(len(G))]
    for src in G:
        distance, _ = bellman_ford(G, src, len(G))
        for dst in G:
            if src != dst:
                result[src][dst] = distance[dst]
    return result

#g = create_random_graph(10, 20)
#g = weighted_graph(g, 10)

#print("Graph: ", g) 

'''
dijstra_result = allpair_dijkstra(g)
print("Dijkstra result: ", dijstra_result)
bellman_result = allpair_bellman_ford(g)
print("Bellman-Ford result: ", bellman_result)

print("Check if the results are the same: ", dijstra_result == bellman_result)
'''

#part 4
def euclidean_distance(station1_id, station2_id, coordinates_dict):
    coord1 = coordinates_dict[station1_id]
    coord2 = coordinates_dict[station2_id]
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def A_Star(graph, source, destination, heuristic):
    
    open_list = [(0, source)]  #to record the node 
    closed_list = set() #to check visited node

    g_values = {node: float('inf') for node in range(len(graph))} #set every cost to infinity \
    g_values[source] = 0

    parent = {source: None}

    while open_list:
        current_f, current_node = heapq.heappop(open_list) # variable to receive the smallest value in the list
        
        if current_node == destination:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = parent[current_node]
            return path[::-1] 

        closed_list.add(current_node)


        for neighbor_raw in graph[current_node]:
            neighbor = int(neighbor_raw)  

            if neighbor in closed_list:
                continue

            if neighbor not in g_values:
                g_values[neighbor] = float('inf')
        
            cost = graph[current_node][neighbor]['weight']
            tentative_g = g_values[current_node] + cost

            if tentative_g < g_values[neighbor]:
                g_values[neighbor] = tentative_g
                f_value = tentative_g + heuristic(neighbor)
                heapq.heappush(open_list, (f_value, neighbor))
                parent[neighbor] = current_node

    return None  #if no path found

#part 5
#read the csv file
df1 = pd.read_csv('london_connections.csv')

df2 = pd.read_csv('london_stations.csv')

#create the weighted undirected graph
G = nx.Graph()

for _, row in df1.iterrows():
    G.add_edge(row['station1'], row['station2'], weight=row['time'], line = row['line'])

station_coordinates = {
    row['id']: (row['latitude'], row['longitude']) for _, row in df2.iterrows()
}


#print(station_coordinates)
#print(G)
#euclidean_distance(1,100,station_coordinates)

def count_line_changes(path, graph):
    lines_used = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        line = graph[u][v]['line']
        lines_used.append(line)
    return len(set(lines_used)) 
    
def compare_algorithms(graph, source, target, coordinates_dict):

    # Dijkstra alogrithm
    start_d = time.time()
    dist_d, path_d = dijkstra(graph, source, k=len(graph) - 1) 
    end_d = time.time()
    dijkstra_time = end_d - start_d
    dijkstra_distance = dist_d.get(target, float('inf'))

    # A* algorithm
    heuristic = lambda current_id: euclidean_distance(current_id, target, coordinates_dict)
    start_a = time.time()
    path_a = A_Star(graph, source, target, heuristic)
    end_a = time.time()
    astar_time = end_a - start_a
    if path_a and len(path_a) > 1:
        astar_distance = 0
        for i in range(len(path_a) - 1):
            u = path_a[i]
            v = path_a[i + 1]
            astar_distance += graph[u][v]['weight']  
    else:
        astar_distance = float('inf')  

    line_count = count_line_changes(path_a, graph)
    
    return {
        "source": source,
        "target": target,
        "dijkstra_time": dijkstra_time,
        "astar_time": astar_time,
        "dijkstra_distance": dijkstra_distance,
        "astar_distance": astar_distance,
        "dijkstra_path": path_d.get(target, []),
        "astar_path": path_a,
        "line_changes": line_count
    }


all_stations = list(station_coordinates.keys())
results = []
#the experiment that try 20 pairs of random stations 
for _ in range(20):  
    s, t = random.sample(all_stations, 2)
    result = compare_algorithms(G, s, t, station_coordinates)
    results.append(result)


#plot for diagram
df = pd.DataFrame(results)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(df.index, df['dijkstra_time'], label='Dijkstra Time', marker='o')
plt.plot(df.index, df['astar_time'], label='A* Time', marker='x')
plt.xlabel('Test Case Index')
plt.ylabel('Time (seconds)')
plt.title('Dijkstra vs A* Running Time')
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
plt.plot(df.index, df['dijkstra_distance'], label='Dijkstra Distance', marker='o')
plt.plot(df.index, df['astar_distance'], label='A* Distance', marker='x')
plt.xlabel('Test Case Index')
plt.ylabel('Path Distance')
plt.title('Dijkstra vs A* Path Length')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()