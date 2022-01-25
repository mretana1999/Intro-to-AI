##############################################################################
#Assignment A1 - Problem 2: Adversarial search                              ##
#Student Names: Mauricio Retana, Juan Idrovo, Reynaldo Williams             ##
#Due date: 6/11/2021                                                        ##
#Code used as reference: https://www.youtube.com/watch?v=8ext9G7xspg&t=4137s
##############################################################################

import math
import random


class Player():
    def __init__(self, letter):
        self.letter = letter

    def get_move(self, game):
        pass


class HumanPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        valid_square = False
        val = None
        while not valid_square:
            square = input(self.letter + '\'s turn. Input move (0-8): ')
            try:
                val = int(square)
                if val not in game.available_moves():
                    raise ValueError
                valid_square = True
            except ValueError:
                print('Invalid square. Try again.')
        return val


class RandomComputerPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        square = random.choice(game.available_moves())
        return square


class SmartComputerPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        if len(game.available_moves()) == 9:
            square = random.choice(game.available_moves())
        else:
            #initialize alpha to inf and beta to -inf
            square = self.minimax(game, self.letter, -math.inf, math.inf)['position']
        return square

    #modified to implement alpha-beta pruning
    def minimax(self, state, player,alpha, beta):
        max_player = self.letter  # yourself
        if player == 'X':
            other_player = 'O'
        else:
            other_player = 'X'

        # first we want to check if the previous move is a winner 
        #(previous move is always made by other_player so we only check if other_player won)
        if state.current_winner == other_player:
            if other_player == max_player:
                return {'position': None, 'score': 1} #If the agent (AI) won, return a score of 1
            else:
                return {'position': None, 'score': -1} #If opponent won, return a score of -1

        #Check if previous move was a draw
        elif not state.empty_squares():
            return {'position': None, 'score': 0}

        if player == max_player:
            best = {'position': None, 'score': -math.inf}  #each score should maximize
        else:
            best = {'position': None, 'score': math.inf}  #each score should minimize
        for possible_move in state.available_moves():
            state.make_move(possible_move, player)
            sim_score = self.minimax(state, other_player, alpha, beta)  #simulate a game after making that move

            # undo move
            state.board[possible_move] = ' '
            state.current_winner = None
            sim_score['position'] = possible_move  # this represents the move optimal next move

            if player == max_player:  # O is maximazing player since Agent plays as O
                if sim_score['score'] > best['score']:
                    best = sim_score
                if best['score'] > alpha:
                    alpha = best['score']
            else:
                if sim_score['score'] < best['score']:
                    best = sim_score
                if best['score'] < beta:
                    beta = best['score']
            if alpha >= beta:
                break
        return best