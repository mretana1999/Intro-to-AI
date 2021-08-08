###########################################################################
#Assignment A1 - Problem 1: Shortest route                                #
#Student Names: Mauricio Retana, Juan Idrovo, Reynaldo Williams           #
#Due date: 6/11/2021                                                      #
###########################################################################
import math
from queue import Queue
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt


city_names = { #Dictionary used for output purposes only
    'A': 'Arad',
    'B': 'Bucharest',
    'C': 'Craiova',
    'D': 'Dobreta',
    'E': 'Eforie',
    'F': 'Fagaras',
    'H': 'Hirsova',
    'I': 'Iasi',
    'L': 'Lugoj',
    'O': 'Oradea',
    'P': 'Pitesti',
    'R': 'Rimnicu Vilcea',
    'U': 'Urziceni',
    'Z': 'Zerind',
    'S': 'Sibiu',
    'T': 'Timiasoara',
    'G': 'Giurgiu',
    'M': 'Mehadia',
    'V': 'Vaslui',
    'N': 'Neamt'
}

cities_h = { #Dictionary that contains the heuristics used in A* search
    "A": 366,
    "B": 0,
    "C": 160,
    "D": 242,
    "E": 161,
    "F": 178,
    "G": 77,
    "H": 151,
    "I": 226,
    "L": 244,
    "M": 241,
    "N": 234,
    "O": 380,
    "P": 98,
    "R": 193,
    "S": 253,
    "T": 329,
    "U": 80,
    "V": 199,
    "Z": 374,
}

cities = { #Dictionary of dictionaries (weighted adjacency list) that represents the map of Romania as an undirected graph
    'A': {'Z': 75, 'S': 140, 'T': 118} ,
    'B': {'U': 85, 'P': 101, 'G': 90, 'F': 211} ,
    'C': {'D': 120, 'R': 146, 'P': 138} ,
    'D': {'M': 75, 'C': 120} ,
    'E': {'H': 86} ,
    'F': {'S': 99, 'B': 211} ,
    'H': {'U': 98, 'E': 86} ,
    'I': {'V': 92, 'N': 87} ,
    'L': {'T': 111, 'M': 70} ,
    'O': {'Z': 71, 'S': 151} ,
    'P': {'R': 97, 'B': 101, 'C': 138} ,
    'R': {'S': 80, 'C': 146, 'P': 97} ,
    'U': {'V': 142, 'B': 85, 'H': 98} ,
    'Z': {'A': 75, 'O': 71} ,
    'S': {'A': 140, 'F': 99, 'O': 151, 'R': 80} ,
    'T': {'A': 118, 'L': 111} ,
    'G': {'B': 90} ,
    'M': {'D': 75, 'L': 70} ,
    'V': {'I': 92, 'U': 142},
    'N': {'I': 87}
}

city_coordinates =   { #dictionary with coordinates for networkx plot functionality
                'A': (91, 492), 'B': (400, 327), 'C': (253, 288), 'D': (165, 299), 'E': (562, 293), 'F': (305, 449), 'G': (375, 270), 'H': (534, 350), 'I': (473, 506), 'L': (165, 379), 'M': (168, 339), 'N': (406, 537), 'O': (131, 571), 'P': (320, 368),
                'R': (233, 410), 'S': (207, 457), 'T': (94, 410), 'U': (456, 350), 'V': (509, 444), 'Z': (108, 531)
                }


def bfs(start, destination):
    '''Finds and returns path with the fewest amount of nodes using depth first search algorithm'''
    queue_FIFO = [start] #We used a python list with pop(0) for FIFO queue functionality
    parents = {} #dictionary that will store list of parents. Will be used to find path.
    parents[start] = None
    visited = [start] #mark start node as visited
    num_visited = 1
    found_flag = False
    while queue_FIFO:
        current_point = queue_FIFO.pop(0)
        if current_point == destination:
            print('Number of visited (explored) nodes: ', num_visited)
            found_flag = True
            break
        for neighbor in sorted(cities[current_point]): #sort neighbors to be visited alphabetically
            if neighbor not in visited:
                num_visited += 1
                parents[neighbor] = current_point
                visited.append(neighbor)
                queue_FIFO.append(neighbor)                
    if not found_flag:
        return None
    path = [destination]
    next_parent = parents[destination]
    while next_parent is not None:
        path.append(next_parent)
        next_parent = parents[next_parent]
    path.reverse()
    return path

def dfs(start, destination):
    '''Finds and returns first found path using depth first search algorithm'''
    stack = [start] #stack is a python default list, initialize it with start node
    parents = {}
    parents[start] = None
    visited = [start] #mark start node as visited
    num_visited = 1
    found_flag = False
    
    while stack:
        current_point = stack.pop()
        if current_point == destination:
            print('Number of visited (explored) nodes: ', num_visited)
            found_flag = True
            break
        for neighbor in sorted(cities[current_point], reverse=False): #Sort list alphabetically in reverse so that last item in stack is the first letter
            if neighbor not in visited:
                num_visited += 1
                parents[neighbor] = current_point
                visited.append(neighbor)
                stack.append(neighbor)                
    if not found_flag:
        return None
    path = [destination]
    next_parent = parents[destination]
    while next_parent is not None:
        path.append(next_parent)
        next_parent = parents[next_parent]
    path.reverse()
    return path

def print_solution(path):
    '''Print all the cities visited and their distances as well as total distance given a path'''
    total_distance = 0
    #Get the city initial from each item and the next in path list
    for index, node in enumerate(path):
        if index < len(path) - 1:
            city1, city2 = str(node), str(path[index + 1])
            distance = cities[city1][city2] #get distance from city 1 to city 2 from edge weights in cities list
            print("Distance from",city1,'to',city2,'is:',distance)
            total_distance += distance
    print('The total distance is:',total_distance,"km")

def sort_function(item):
    return item['priority']

def a_star(start, destination):
    '''Finds and returns path with the shortest distance using A* search'''
    priority_queue = [{'name': start, 'priority':0}] #priority value = distance + heuristic. This value is represented as f in f(n) = g(n) + h(n)
    parents = {}
    parents[start] = None
    cost_so_far = {} #Cost from root to current point
    cost_so_far[start] = 0
    num_visited = 1
    found_flag = False

    while priority_queue:
        priority_queue.sort(key = sort_function) #sort queue by value of f(n) in ascending order
        current_point = priority_queue.pop(0)['name'] #pop the node with the lowest current cost
        if current_point == destination:
            found_flag = True
            break

        for neighbor in cities[current_point]: #cities is the dictionary defined globally above
            new_cost = cost_so_far[current_point] + cities[current_point][neighbor] 
            num_visited += 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]: #if neighbor has not been visited or new f(n) < current f(n)
                cost_so_far[neighbor] = new_cost
                priority = cost_so_far[neighbor] + cities_h[neighbor] #f(n) = g(n) + h(n)
                priority_queue.append({'name': neighbor, 'priority':priority})
                parents[neighbor] = current_point
    if not found_flag:
        return None
    path = [destination]
    next_parent = parents[destination]
    while next_parent is not None:
        path.append(next_parent)
        next_parent = parents[next_parent]
    path.reverse()
    #print('Total distance is:', cost_so_far[destination],'km')
    print('Number of visited (explored) nodes is:',num_visited)
    return path
#---------------------------------------Main-----------------------------------------------------------------------------------#
#Create graph from adjacency list using Networkx library and matplotlib

gr = {
    from_: {
        to_: {'weight': w}
        for to_, w in to_nodes.items()
    }
    for from_, to_nodes in cities.items()
}

G = nx.from_dict_of_dicts(gr)
G.edges.data('weight')
pos = city_coordinates
nx.draw(G, pos, with_labels=True)
labels = {e: G.edges[e]['weight'] for e in G.edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show(block=False)
plt.pause(2)
#plt.close()


destination = 'B' #destination is hard-coded as Bucharest B
start = ''
print('\nThe purpose of this program is to implement uninformed search (Breadth First S. and Depth Firs S.) as well as heuristic search (A*) algorithms to solve the classical problem')
print('of finding the best route between two cities applied to the map of Romania.')
print('\nThe following is a list of the cities you can depart from:',city_names)

while True:
    start = input('\nEnter the first letter of the city you plan to depart from or [X] to exit: ')
    start = start[0].upper()
    if start not in city_names and start != 'X':
        print('City not found, try again')
        continue
    elif start == 'X':
        break
    else:
        selection = input('Choose an algorithm: A*[A] / BFS[B] / DFS[D] or Exit[X]: ')
        selection = selection[0].upper()
        print('Your starting city is',start,'whose name is', city_names[start])
        print('Your destination city is',destination,'whose name is', city_names[destination])
        if selection == 'A':
            path = a_star(start, destination)
            print('A* Search path (itinerary) from start to destination is:', path)
            print_solution(path)

        elif selection == 'B':
            path = bfs(start,destination)
            print('Breadth First Search path (itinerary) from start to destination is: ',path)
            print_solution(path)

        elif selection == 'D':
            path = dfs(start, destination)
            print('Depth First Search path (itinerary) from start to destination is: ', path)
            print_solution(path)
        else:
            break

print('Thank you, goodbye!')