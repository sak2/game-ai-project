'''

COMP30024 Project Player module 
Semester 1 2018
Tamara Hogan (682452) & Saleh Ahmed Khan (798838)

'''

from game2 import Game
from node import Node
import random
import datetime
from copy import deepcopy
from math import log,sqrt

# piece tokens
WHITE = 'O'
BLACK = '@'

# chance of making a random move 
RANDOM_CHOICE = 0

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
        elif x > RANDOM_CHOICE:
            # if move is to be non-random perform minimax search
            monte_carlo = MonteCarlo(self.game)
            move = monte_carlo.get_play()
        
        # apply the chosen move to the game  
        self.game.make_move(move)
        
        # return the move taken 
        return move
    
    def update(self, action):
        # apply the opponents move to the game 
        self.game.make_move(action)
        

            
class MonteCarlo(object):
    '''
    Based on 
    https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/
    '''
    def __init__(self, game, **kwargs):
        
        self.game = game
        self.states = [game]
        
        self.wins = {}
        self.plays = {}
        
        seconds = kwargs.get('time', 0.1)
        self.calculation_time = datetime.timedelta(seconds=seconds)
        self.max_moves = kwargs.get('max_moves', 200)
        self.C = kwargs.get('C', 1.4)
        
    def update(self, state):
        self.states.append(state)
        
    def get_play(self):
        self.max_depth = 0
        state = self.states[-1]
        player = self.game.player
        legal = self.game.moves()
        
        if not legal:
            return None
        if isinstance(legal[0], int):
            return legal
        
        games = 0
        
        begin = datetime.datetime.utcnow()
        
        while datetime.datetime.utcnow() - begin < self.calculation_time:
            self.run_simulation()
            games += 1
        
        
        move_states = []
        
        for move in legal:
            new_state = deepcopy(state)
            new_state.make_move(move)
            move_states.append((move,convert_to_list(new_state.board)))
        
        #print("Games: " + str(games))
        #print("Time elapsed: " + str(datetime.datetime.utcnow() - begin))
        
        percent_wins, move = max((self.wins.get((player, S), 0) /self.plays.get((player, S), 1),p)
            for p, S in move_states)
        for x in sorted(
            ((100 * self.wins.get((player, S), 0) /
              self.plays.get((player, S), 1),
              self.wins.get((player, S), 0),
              self.plays.get((player, S), 0), p)
             for p, S in move_states),
            reverse=True
        ):
            x = 2
            #print("{3}: {0:.2f}% ({1} / {2})".format(*x)) 
        #print("Maximum search depth reached: " + str(self.max_depth))
        return move 
    def run_simulation(self):
        plays, wins = self.plays, self.wins
        
        visited_states = set()
        state = deepcopy(self.game)
        player = state.player
        
        expand = True
                
        for t in range(1,self.max_moves+1):
            
            legal = state.moves()
            
            if legal == []:
                move = None
                state.make_move(move)
            elif isinstance(legal[0],int):
                move = legal
                state.make_move(move)
            elif isinstance(legal,list):
                move_states = []
                
                #for move in legal: 
                #   new_state = deepcopy(state)
                #  new_state.make_move(move)
                # new_board = convert_to_list(new_state.board)
                #move_states.append((move,new_board))
                
                #all_info = all(plays.get((player,S)) for p,S in move_states)
                if False:
                    log_total = log(sum(plays[(player,S)] for p,S in move_states))
                    _, move, board = max(((wins[(player, S)] / plays[(player, S)]) + self.C * sqrt(log_total / plays[(player, S)]), p, S) for p, S in move_states)
                    state.make_move(state)
                elif True:
                    move = random.choice(legal)
                    state.make_move(move)
            
            
            board = convert_to_list(state.board)
            
            if expand and (player,board) not in self.plays:
                expand = False
                self.plays[(player,board)] = 0
                self.wins[(player,board)] = 0
                if t > self.max_depth:
                    self.max_depth = t
                
            visited_states.add((player,board))
            
            player = state.player
            
            goal = state.check_goal()
            
            if goal[0] and state.turns >=24:
                break
        
        # THIS IS A PROBLEM
        # DICTIONARIES ARE NOT HASHABLE
        for player,state in visited_states:
            if (player,state) not in self.plays:
                continue
            self.plays[(player,state)] += 1
            if player == goal[1]:
                self.wins[(player,state)] += 1
            
def convert_to_list(board):
    board_list = []
    for i in range(8):
        row = []
        for j in range(8):
            row.append(board[(j,i)])
        board_list.append(row)
    return tuple(tuple(x) for x in board_list)