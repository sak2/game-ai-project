'''

COMP30024 Project Player module
Semester 1 2018
Tamara Hogan (682452) & Saleh Ahmed Khan (798838)

'''

from game2 import Game
from node import Node
import random
from copy import deepcopy

from partA import game
from selfplay import self_play

# piece tokens
WHITE = 'O'
BLACK = '@'
EMPTY = '-'
CORNER = 'X'
DEAD = 'D'

# chance of making a random move
RANDOM_CHOICE = 1

# depth limit for minimax
DEPTH_LIMIT = 12

# infinity float
INF = float('inf')

class Minimax(object):

    '''
    Defines a module that plays Watch Your Back! in accordance
    with the Part B project specification.

    Attributes:
        colour ('char'): token of the player
        opponent ('char'): token of the opponent
        game ('Game'): current state of the game

    Functions:
        action(self,turns): returns next move for the player
        update(self,action): updates game according to opponent's action
    '''

    def __init__(self, colour):
        '''
        Parameters:
            colour ('char'): token being played by the player module
        '''

        if colour == 'black':
            self.opponent = WHITE
            self.colour = BLACK
        elif colour == 'white':
            self.opponent = BLACK
            self.colour = WHITE

        self.game = Game()


    def action(self, turns):

        # generate all legal moves for current game state
        move_list = self.game.moves()

        # if there is no legal moves, forfeit turn
        if move_list == []:
            return None

        # get the utility for every possible move
        utilities = []
        for move in move_list:

            utility = self.utility(move)
            utilities.append((utility, move))
            temp_game = deepcopy(self.game)
            temp_game.make_move(move)

        # apply the chosen move to the game
        max_u = utilities[0][0]
        move = utilities[0][1]
        for (utility, move) in utilities:
            if utility > max_u:
                max_u = utility
                move = move
        self.game.make_move(move)
        # return the move taken
        return move

    def update(self, action):
        # apply the opponents move to the game
        self.game.make_move(action)

    # def minimax(self, move_list):
    #     for i in move_list:
            
    def utility(self, last_move):
        
        '''
        Finds whose player gets eliminated    
        '''
        eval_game = deepcopy(self.game)

        eval_game.make_move(last_move)
        if self.colour == WHITE:
            return eval_game.count_pieces()[0]
        else:
            return eval_game.count_pieces()[1]
