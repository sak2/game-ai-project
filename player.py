'''

COMP30024 Project Player module 
Semester 1 2018
Tamara Hogan (682452) & Saleh Ahmed Khan (798838)

'''

from game2 import Game
from node import Node
import random
from selfplay import self_play

# piece tokens
WHITE = 'O'
BLACK = '@'

# chance of making a random move 
RANDOM_CHOICE = 1

# depth limit for minimax 
DEPTH_LIMIT = 12

# infinity float
INF = float('inf')

class Player(object):
    
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
        
        # generate a random number between 0 and 1 
        x = random.uniform(0,1)
        
        # if x is under our chance of a random choice make a random move 
        if x <= RANDOM_CHOICE:
            move = random.choice(move_list)
        
        # apply the chosen move to the game  
        self.game.make_move(move)
        
        # return the move taken 
        return move
    
    def update(self, action):
        # apply the opponents move to the game 
        self.game.make_move(action)
        

