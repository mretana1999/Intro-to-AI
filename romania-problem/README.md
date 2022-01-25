##############################################################################
# Assignment A1 - Problem 1: Shortest route                              #
### Student Names: Mauricio Retana, Juan Idrovo, Reynaldo Williams             ###
### Due date: 6/11/2021                                                        ###
### Code used as reference: https://www.youtube.com/watch?v=8ext9G7xspg&t=4137s ###
##############################################################################

## Purpose:
The purpose of this program is to implement uninformed search (Breadth First S. and Depth Firs S.) as well as heuristic search (A*) algorithms to solve the classical problem
of finding the best route between two cities applied to the map of Romania.

## Libraries:
This program uses networkx, matplotlib, and pandas libraries. Please make sure to install the latest versions of these libraries in order to run the program.
These libraries are used to show the map of Romania for 2 seconds when the program is run. The program will continue running after 2 seconds.

## How to run:
Run the program on your installed IDE or in your command terminal. Please note that the map of Romania will be displayed when the program is run for 2 seconds. Then the program will continue running. The user decides whether they want to close the window with the drawing of the map. After 2 seconds, the user is prompted to enter the initial of the name they want to depart from. A list of the cities that the user can choose is displayed after the map of Romania closes. The destination is hard-coded as Bucharest. 

Once the user has chosen the starting city, they are prompted to select which algorithm to perform the search. The selected algorithm is run and the itinerary will be shown with a list of the cities to be visited and the corresponding distances to be travelled.



