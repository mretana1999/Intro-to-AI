##############################################################################
# Assignment A1 - Problem 2: Adversarial search                              #
### Student Names: Mauricio Retana, Juan Idrovo, Reynaldo Williams             ###
### Due date: 6/11/2021                                                        ###
### Code used as reference: https://www.youtube.com/watch?v=8ext9G7xspg&t=4137s ###
##############################################################################

## Purpose:
The purpose of this program is to implement the Min-max algorithm to create an AI agent to play against a human which always seeks the most optimal score.
This tic-tac-toe also implements Alpha-beta pruning, which is a technique used by the agent to ignore those paths that lead to repetitive scores. This makes the Min-max algorithm less computationally expensive to be implemented.

## How to play:
To play the tic-tac-toe game, run the a1_mretanarodri2018_game.py file. Please make sure that player.py and a1_mretanarodri2018_game.py are in the same directory since player.py contains the class definitions for the Human player and Agent player. The file player.py also contains the Min-max (or minimax) algorithm implementation. Note that this implementation includes the alpha-beta pruning technique that is explained in the textbook by Rishal Hurbans. 

As stated in the instructions, the user always plays as 'X' and thus always plays their move first. The AI agent will control the positions of the O. The game finishes until there is a winner or there are no more spaces to play (tie). The user is then asked if they want to play again or not.

## Libraries:
This program uses the math and time libraries which come with Python 3. by default.