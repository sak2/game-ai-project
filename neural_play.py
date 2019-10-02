'''
COMP30024 Project Player module 
Semester 1 2018
Tamara Hogan (682452) & Saleh Ahmed Khan (798838)
'''

from game2 import Game
from node import Node
import random
from neuralnet import Neural_Network
from feature_conversion import convert_to_feature
from copy import deepcopy

# piece tokens
WHITE = 'O'
BLACK = '@'

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
        
        self.net = Neural_Network('neural.csv')
        
        self.epsilon = 1
        
        if self.colour == WHITE:
            self.idx = 0
        else:
            self.idx = 1
        
        
    
    def action(self, turns):
        
        # generate all legal moves for current game state        
        move_list = self.game.moves()
        
        # if there is no legal moves, forfeit turn
        if move_list == []:
            return None
        
        # generate a random number between 0 and 1 
        x = random.uniform(0,1)
        
        # if x is under our chance of a random choice make a random move 
        if x < 1 - self.epsilon:
            move = random.choice(move_list)
        else: 
            move = self.pick(move_list)
        
        # apply the chosen move to the game  
        self.game.make_move(move)
        
        # return the move taken 
        return move
    
    def pick(self, moves):
        
        root = Node(self.game)
        best_win_child, win_v, _ = minimax(root, self, self.idx, 18, None)
        best_child = best_win_child
        
        return best_child.path[0]
    
    def update(self, action):
        # apply the opponents move to the game 
        self.game.make_move(action)
    
    def self_play(self):
        
        own_game = Game()
        
        game_states = [deepcopy(own_game)]
        
        reward = own_game.reward()
        
        while not reward:
            
            moves = own_game.moves()
            
            if own_game.player == WHITE:
                self.idx = 0
            else:
                self.idx = 2
            
            move = self.pick(moves)
        
            own_game.make_move(move)
            
            game_states.append(deepcopy(own_game))
            
            reward = own_game.reward()
        return game_states, reward
    
def minimax(root, player, index, depth=4,cutoff_test=None):
    '''
    Based on alpha beta cutoff search from
    https://github.com/aimacode/aima-python/blob/master/games.py
    
    Returns the best child node of the root. 
    
    Parameters:
        root ('Node'): the root of the search tree
        depth ('int'): the depth cut off for the search
        cutoff_test ('func'): determines when a terminal node or the depth cutoff 
                                is reached
        eval_fun ('func'): scoring function for nodes 
        
    '''    
    def max_value(root, player, index, alpha, beta, depth):
        if cutoff_test(depth):
            features = convert_to_feature(root.game)
            result = root.game.check_goal()
            if result[0]:
                return root.game.reward()[index], root
            else:
                return player.net.forward_prop(features)['a2'][0][index], root 
            
        
        v = -INF
        
        if not root.expanded:
            root.expand()
        node = None
        for child in root.children:
            new_v, new_node = min_value(child,player,index, alpha,beta,depth+1)
            if new_v > v:
                node = new_node
                v = new_v
            if v >= beta:
                return v, node
            alpha = max(alpha,v)
        
        return v, node
    
    
    def min_value(root, player,index, alpha, beta, depth):
        if cutoff_test(depth):
            features = convert_to_feature(root.game)
            result = root.game.check_goal()
            if result[0]:
                return root.game.reward()[index], root
            else:
                return player.net.forward_prop(features)['a2'][0][index], root 
        v = INF
        node = None
        if not root.expanded:
            root.expand()
        for child in root.children:
            new_v, new_node = max_value(child,player, index,alpha,beta,depth+1)
            if new_v < v:
                node = new_node
                v = new_v
            if v <= alpha:
                return v, node
            beta = min(beta,v)
        return v, node
    
    cutoff_test = (lambda d: depth > d or root.game.check_goal()[0])
    
    best_score = -INF
    beta = INF
    best_action = None
    best_node = None
    
    if not root.expanded:
        root.expand()
    
    for child in root.children:
        v, node = min_value(child, player, index, best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = child 
            best_node = node
    return best_action, v, best_node