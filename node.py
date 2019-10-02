'''

COMP30024 Project Node module 
Semester 1 2018
Tamara Hogan (682452) & Saleh Ahmed Khan (798838)

'''

from copy import deepcopy


WHITE = 'O'
BLACK = '@'
CORNER = 'X'
EMPTY = '-'
DEAD = 'D'

class Node(object):
    
    '''
    
    Derived from the Node class in the AIMA code
    found at 
    https://github.com/aimacode/aima-python/blob/master/search.py
    
    Defines a node for a search tree data structure. 
    
    Stores all necessary information required for variety of search
    strategies, including path taken to current node. 
    
    Attributes: 
        game ('Game'): game associated with Node
        state ('list' of 'list' of 'char'): board configuration
        parent ('Node'):  parent node
        path ('list' of 'tuple' of ('int','int')): moves taken from root to node
        children ('list' of 'Node'): child nodes 
        depth ('int'): level of tree where node is located 
        g ('int'): path cost 
        h ('int'): A* hueristic 
        f ('int'): A* estimated path cost 
        expanded ('Bool'): whether search tree has been expanded at node
    
    '''
    
    def __init__(self,game,parent=None,path=None,children=None,depth=0):
        '''
        
        Parameters: 
            game ('Game'): game associated with node 
            state ('list' of 'list' of 'char'): board configuration
            parent ('Node' or None): parent node
            path ('list' of 'tuple' of ('int,'int')): moves taken from root 
            children ('list' of 'Node'): child nodes
            depth ('int'): level of search tree where node is located
            
        '''

        # parent node
        self.parent = parent 
        # game that node belongs to 
        self.game = game
        # path from root to node 
        self.path = [] if path is None else path
        # children of node 
        self.children = []
        
        # generate children if available 
        if children is not None:
            for child in children:
                self.add_child(self.game.player, child)
        
        # depth in search tree 
        self.depth = depth
        
        # whether node has been expanded in search tree 
        self.expanded = False 
        
        
        
    
    
    def __eq__(self, other):
        return self.game.board == other.game.board
        
    def expand(self):
        '''
        
        Expands search tree at node to include all available child nodes. 
        
        '''
        
        # generate all moves available from current state 
        moves = self.game.moves()
        
        if len(moves) == 0:
            self.add_child()
        else:
            for move in moves:
                self.add_child(move)
        
        self.expanded = True 
    
    
    def add_child(self, move = None):
        '''
        
        Uses given move to generate appropriate child node and add it to 
        self.children. 
        
        Parameters:
            move ('tuple' of ('int','int)): move to be taken in same format 
            as returned in list in self.moves
        
        '''
        
        new_game = deepcopy(self.game)
        
        new_game.make_move(move)

        
        child = Node(new_game,self,self.path + [move], None, self.depth+1)
        
        # append node to self.children
        self.children.append(child)
    
