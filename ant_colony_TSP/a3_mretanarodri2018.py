import random
import math
import numpy as np
import matplotlib.pyplot as plt

#Set seed for random library functions
random.seed(10)

# Ant Colony Optimization (ACO)
# The Ant Colony Optimization algorithm is inspired by the behavior of ants moving between destinations, dropping
# pheromones and acting on pheromones that they come across. The emergent behavior is ants converging to paths of
# least resistance.

# Create a list of cities in a grid of 200*200 by randomly choosing (x,y) coordinates
CITY_COUNT = 25
city_list = [(int(random.uniform(0,200)),int(random.uniform(0,200))) for i in range(CITY_COUNT)]
# Initialize the 2D matrix for storing distances between cities
city_distances = [[0]*CITY_COUNT for i in range(CITY_COUNT)]
# Fill 2D matrix with distances between each city
for i in range(CITY_COUNT):
    for j in range(i+1,CITY_COUNT):
        city_distances[i][j] = city_distances[j][i] = math.sqrt(pow(city_list[i][0] - city_list[j][0], 2) + pow(city_list[i][1] - city_list[j][1], 2))

# The Ant class encompasses the idea of an ant in the ACO algorithm.
# Ants will move to different cities and leave pheromones behind. Ants will also make a judgement about which
# city to visit next. And lastly, ants will have knowledge about their respective total distance travelled.
# - Memory: In the ACO algorithm, this is the list of cities already visited.
# - Best fitness: The shortest total distance travelled across all cities.
# - Action: Choose the next destination to visit and drop pheromones along the way.
class Ant:

    # The ant is initialized to a random city with no previously visited cities
    def __init__(self):
        self.visited_cities = []
        self.visited_cities.append(random.randint(0, CITY_COUNT - 1))

    # Select an city using a random chance or ACO function
    def visit_city(self, pheromone_trails):
        if random.random() < RANDOM_CITY_FACTOR:
            self.visited_cities.append(self.visit_random_city())
        else:
            if USER_SELECTED_OPTIONS_DICT['selection'] == '1':
                self.visited_cities.append(self.roulette_wheel_selection(self.visit_probabilistic_city(pheromone_trails)))
            elif USER_SELECTED_OPTIONS_DICT['selection'] == '2': 
                self.visited_cities.append(self.ranking_selection(self.visit_probabilistic_city(pheromone_trails)))

    # Select an city using a random chance
    def visit_random_city(self):
        all_cities = set(range(0, CITY_COUNT))
        possible_cities = all_cities - set(self.visited_cities)
        return random.randint(0, len(possible_cities) - 1)

    # Calculate probabilities of visiting adjacent unvisited cities
    def visit_probabilistic_city(self, pheromone_trails):
        current_city = self.visited_cities[-1]
        all_cities = set(range(0, CITY_COUNT))
        possible_cities = all_cities - set(self.visited_cities)
        possible_indexes = []
        possible_probabilities = []
        total_probabilities = 0
        for city in possible_cities:
            possible_indexes.append(city)
            pheromones_on_path = math.pow(pheromone_trails[current_city][city], ALPHA)
            heuristic_for_path = math.pow(1 / city_distances[current_city][city], BETA)
            probability = pheromones_on_path * heuristic_for_path
            possible_probabilities.append(probability)
            total_probabilities += probability
        possible_probabilities = [probability / total_probabilities for probability in possible_probabilities]
        return [possible_indexes, possible_probabilities, len(possible_cities)]

    def get_visited_cities(self):
        return self.visited_cities

    # Select an city using the probabilities of visiting adjacent unvisited cities
    @staticmethod
    def roulette_wheel_selection(probabilities):
        slices = []
        total = 0
        possible_indexes = probabilities[0]
        possible_probabilities = probabilities[1]
        possible_cities_count = probabilities[2]
        for i in range(0, possible_cities_count):
            slices.append([possible_indexes[i], total, total + possible_probabilities[i]])
            total += possible_probabilities[i]
        spin = random.random()
        result = [s[0] for s in slices if s[1] < spin <= s[2]]
        return result[0]
    
    @staticmethod
    def ranking_selection(probabilities):
        '''Apply linear ranking selection to select next city index'''
        slices = []
        total = 0
        possible_indexes = probabilities[0]
        fitness = probabilities[1]
        num_ranks = probabilities[2]
        ranked_possible_indexes = list(zip(possible_indexes,fitness))
        #Get indexes of sorted list of (index,fitness) tuples -sorted by fitness in ascending order-
        ranked_possible_indexes = [element[0] for element in sorted(ranked_possible_indexes,key = lambda element_:element_[1])]
        sum_all_ranks = sum(range(1,num_ranks + 1))
        possible_probabilities = [i/sum_all_ranks for i in range(1,num_ranks + 1)]
        #sum_possible_probabilities = sum(possible_probabilities)
        for i in range(0, num_ranks):
            slices.append([ranked_possible_indexes[i],total,total + possible_probabilities[i]])
            total += possible_probabilities[i]
        spin = random.random()
        result = [s[0] for s in slices if s[1] < spin <= s[2]]
        return result[0]

    # Get the total distance travelled by this ant
    def get_distance_travelled(self):
        total_distance = 0
        for a in range(1, len(self.visited_cities)):
            total_distance += city_distances[self.visited_cities[a]][self.visited_cities[a-1]]
        total_distance += city_distances[self.visited_cities[0]][self.visited_cities[len(self.visited_cities) - 1]]
        return total_distance

    def print_info(self):
        print('Ant ', self.__hash__())
        print('Total cities: ', len(self.visited_cities))
        print('Total distance: ', self.get_distance_travelled())


# The ACO class encompasses the functions for the ACO algorithm consisting of many ants and cities to visit
# The general lifecycle of an ant colony optimization algorithm is as follows:

# - Initialize the pheromone trails: This involves creating the concept of pheromone trails between cities
# and initializing their intensity values.

# - Setup the population of ants: This involves creating a population of ants where each ant starts at a different
# city.

# - Choose the next visit for each ant: This involves choosing the next city to visit for each ant. This will
# happen until each ant has visited all cities exactly once.

# - Update the pheromone trails: This involves updating the intensity of pheromone trails based on the ants’ movements
# on them as well as factoring in evaporation of pheromones.

# - Update the best solution: This involves updating the best solution given the total distance covered by each ant.

# - Determine stopping criteria: The process of ants visiting cities repeats for a number of iterations. One
# iteration is every ant visiting all cities exactly once. The stopping criteria determines the total number of
# iterations to run. More iterations will allow ants to make better decisions based on the pheromone trails.
class ACO:

    def __init__(self, number_of_ants_factor):
        self.number_of_ants_factor = number_of_ants_factor
        # Initialize the array for storing ants
        self.ant_colony = []
        # Initialize the 2D matrix for pheromone trails
        self.pheromone_trails = []
        # Initialize the best distance in swarm
        self.best_distance = math.inf
        self.best_ant = None
        self.progress = [math.inf]
        self.best_route = []

    # Initialize ants at random starting locations
    def setup_ants(self, number_of_ants_factor):
        number_of_ants = round(CITY_COUNT * number_of_ants_factor)
        self.ant_colony.clear()
        for i in range(0, number_of_ants):
            self.ant_colony.append(Ant())

    # Initialize pheromone trails between cities
    def setup_pheromones(self):
        for r in range(0, len(city_distances)):
            pheromone_list = []
            for i in range(0, len(city_distances)):
                pheromone_list.append(1)
            self.pheromone_trails.append(pheromone_list)

    # Move all ants to a new city
    def move_ants(self, ant_population):
        for ant in ant_population:
            ant.visit_city(self.pheromone_trails)

    # Determine the best ant in the colony - after one tour of all cities
    def get_best(self, ant_population):
        for ant in ant_population:
            distance_travelled = ant.get_distance_travelled()
            if distance_travelled < self.best_distance:
                self.best_distance = distance_travelled
                self.best_ant = ant
        return self.best_ant

    # Update pheromone trails based ant movements - after one tour of all cities
    def update_pheromones(self, evaporation_rate):
        for x in range(0, CITY_COUNT):
            for y in range(0, CITY_COUNT):
                self.pheromone_trails[x][y] = self.pheromone_trails[x][y] * evaporation_rate
        for ant in self.ant_colony:
            visited_cities_pairs = []
            for i in range(len(ant.visited_cities)):
                visited_cities_pairs.append((ant.visited_cities[i] , ant.visited_cities[(i+1)%len(ant.visited_cities)]))
            for x in range(0, CITY_COUNT):
                for y in range(0, CITY_COUNT):
                    if (x,y) in visited_cities_pairs:
                        self.pheromone_trails[x][y] += 1 / ant.get_distance_travelled()
    
    def update_pheromones_elite(self):
        ''' Adds (1/distance) to the pheromone trails for the cities visited by the best ant '''
        visited_cities_pairs = []
        best_visited_cities = self.best_ant.visited_cities #indexes of cities visited by the best ant
        for i in range(len(best_visited_cities)):
            visited_cities_pairs.append((best_visited_cities[i] , best_visited_cities[(i+1)%len(best_visited_cities)]))
        for x in range(CITY_COUNT):
            for y in range(CITY_COUNT):
                if (x,y) in visited_cities_pairs:
                    self.pheromone_trails[x][y] += 1/self.best_ant.get_distance_travelled()


    # Tie everything together - this is the main loop
    def solve(self, total_iterations, evaporation_rate):
        self.setup_pheromones()
        stagnation_count = 0
        for i in range(0, TOTAL_ITERATIONS):
            self.setup_ants(NUMBER_OF_ANTS_FACTOR) 
            for r in range(0, CITY_COUNT - 1): #when colony is setup, ants have already visited 1 city
                self.move_ants(self.ant_colony)
            self.update_pheromones(evaporation_rate)
            self.best_ant = self.get_best(self.ant_colony)
            if USER_SELECTED_OPTIONS_DICT['pheromone update'] == '2':
                self.update_pheromones_elite()
            self.progress.append(self.best_ant.get_distance_travelled())
            print(i, ' Best distance: ', self.best_ant.get_distance_travelled())
            if USER_SELECTED_OPTIONS_DICT['stop condition'] == '2':
                if self.progress[i+1] <= self.progress[i] * 1.01:
                    stagnation_count += 1
                    if stagnation_count > 30:
                        print('Optimal solution has stagnated, stopping program!')
                        break
                else:
                    stagnation_count = 0

def parse_visited_cities_indexes(visited_cities_indexes):
    """Returns the corresponding xy coordinates given the indexes of the original city_list"""
    # Visited cities' coordinates + initial city coordinates
    return [city_list[index] for index in visited_cities_indexes] + [city_list[visited_cities_indexes[0]]]

def plot_progress(progress):
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.title('Best route across generations')
    plt.show()

def plot_route(route, subplot_title = ""):
    #create and show cities' positions subplots
    x = [coordinate[0] for coordinate in route]
    y = [coordinate[1] for coordinate in route]
    plt.subplot(1,2,1)
    plt.scatter(x,y, color = 'b')
    plt.ylabel('y-coordinates')
    plt.xlabel('x-coordinates')
    plt.title('Position of cities to be visited')
    #create and show current optimal route
    plt.subplot(1,2,2)
    plt.plot(x,y, color = 'r', zorder = 2) #draw connecting lines
    plt.scatter(x,y, color = 'b', zorder = 1) #draw dots
    plt.ylabel('y-coordinates')
    plt.xlabel('x-coordinates')
    plt.title('Best current solution')
    plt.suptitle(subplot_title)
    plt.show()

def pretty_print_route(route):
    ''' Pretty prints route list of tuples'''
    print('The optimal route found is: ', end='')
    for i in range(len(route) - 1):
        if i % 10 == 0 : print()
        print(f"{route[i]} -> ", end='')
    print(route[0])

def get_user_selection():
    for step in OPTIONS_AVAILABLE_DICT:
        print(f"Which option would you like for **{step}** ")
        for option, description in OPTIONS_AVAILABLE_DICT[step].items():
            print(f"Option {option} : {description}")
        while True:
            decision = input(f" Enter your selection for **{step}**: ")
            if decision in OPTIONS_AVAILABLE_DICT[step]:
                USER_SELECTED_OPTIONS_DICT[step] = decision
                break
            else:
                print('Invalid input, try again: ')
                continue
    print('\nYour final selection is: ')
    for step, decision_num in USER_SELECTED_OPTIONS_DICT.items():
        print(f"{step}: {decision_num} {OPTIONS_AVAILABLE_DICT[step][decision_num]}")
    print('Please wait for the program to finish running.')

OPTIONS_AVAILABLE_DICT = {
                            'selection':{'1':'Roulette Wheel Selection','2':'Ranking Selection'},
                            'pheromone update':{'1':'Normal update', '2':'Favor elite ant trails'},
                            'stop condition':{'1':'Specified number of Generations', '2':'Stop if no improvement (Stagnation)'}
                        }

USER_SELECTED_OPTIONS_DICT = {'selection':'1', 'pheromone update':'1', 'stop condition':'1'}

# Set the percentage of ants based on the total number of cities
NUMBER_OF_ANTS_FACTOR = 0.5
# Set the number of tours ants must complete
TOTAL_ITERATIONS = 100
# Set the rate of pheromone evaporation (0.0 - 1.0)
EVAPORATION_RATE = 0.4
# Set the probability of ants choosing a random city to visit (0.0 - 1.0)
RANDOM_CITY_FACTOR = 0.15
# Set the weight for pheromones on path for selection
ALPHA = 4
# Set the weight for heuristic of path for selection
BETA = 7

get_user_selection()
initial_route = city_list + [city_list[0]]
plot_route(initial_route,'Initial Route Before ACO')
aco = ACO(NUMBER_OF_ANTS_FACTOR)
aco.solve(TOTAL_ITERATIONS, EVAPORATION_RATE)
route = parse_visited_cities_indexes(aco.best_ant.get_visited_cities())
pretty_print_route(route)
plot_route(route, 'Best Found Route After ACO')
plot_progress(aco.progress)