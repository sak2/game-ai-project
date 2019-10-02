'''

COMP30024 Project Part A 
Semester 1 2018
Tamara Hogan (682452) & Saleh Ahmed Khan (798838)

'''

from copy import deepcopy
import sys

WHITE = 'O'
BLACK = '@'
EMPTY = '-'
CORNER = 'X'
DOESNT_MATTER = -1

###########################################################################
########################## CLASSES ########################################
###########################################################################

class Game(object):
    '''
    
    Derived from the Problem class in the AIMA code
    found at 
    https://github.com/aimacode/aima-python/blob/master/search.py
    
    Adapted to the specifics of Watch Your Back!
    
    Stores the initial game state (the board configuration) as well
    as the goal number of pieces in the form (# of white, # of black). 
    Game will end if either number is reached. 
    
    Also provides functions for all required game functions. 
    
    Attributes:
        initial ('list' of 'list' of 'char'): initial board configuration
        goal ('tuple' of 'int'): goal piece numbers 
		h_factor ('int'): defines a constant factor our admissible heuristic
		will be multiplied by to increase running speed 
		white ('int'): number of white pieces on the initial board
		black ('int'): number of black pieces on the initial board 
    
    '''
    def __init__(self, initial, goal = None):
        
        '''
        
        Parameters: 
            initial ('list' of 'list' of 'char'):  initial board configuration
            goal ('tuple' of 'int' or None):  goal state for game 
            
        '''
        
        self.initial = initial
        
        # gives initial goal to disregard white pieces and desire 
        # 0 black pieces if no other goal is given
        self.goal = (DOESNT_MATTER,0) if goal is None else goal
        
        self.h_factor = 1
        
        self.white = self.count_pieces(self.initial)[0]
        self.black = self.count_pieces(self.initial)[1]
        
        
        # factor the heuristic based on the ratio of pieces on the board 
        # this increases efficiency 
        if abs(self.white - self.black) <= 1 :
            self.h_factor = 3
        elif self.white > self.black:
            self.h_factor = 3*self.white/self.black
        elif self.black > self.white:
            self.h_factor = 3*self.black/self.white
        
    def count_pieces(self, state):
        '''
        
        Counts number of player pieces left on the game board. 
        
        Parameters: 
            state ('list' of 'list' of 'char'): board configuration
            
        Returns: 
            ('int', 'int'): number of white pieces in first space followed by
            number of black pieces 
            
        '''
        white = 0
        black = 0
        
        # iterate over the board counting any pieces 
        for i in range(0,8):
            for j in range(0,8):
                if state[i][j] == WHITE:
                    white += 1
                if state[i][j] == BLACK:
                    black += 1
        
        # return totals 
        return (white,black) 
    
    def moves(self, state, player):
        '''
        
        Generates all available moves for the given player on the current 
        board configuration. 
        
        Parameters: 
            state ('list' of 'list' of 'char'): board configuration
            player ('char'): player to generate moves for 
        
        Returns: 
            'list' of (('int,'int'),('int','int')): all available moves in the 
            format of the initial position of the piece followed by the new 
            position of the piece 
        
        '''
        # initialize empty list of moves 
        moves = []
        
        # iterate over the board 
        for i in range(0,8):
            for j in range(0,8):
                
                # if a space contains given player, we check for moves 
                if state[i][j] == player:
                    
                    # check if piece can move up 
                    if i > 0 and state[i-1][j] == EMPTY:
                        moves.append(((j,i),(j,i-1)))
                    elif (i > 1 and state[i-1][j] != EMPTY 
                          and state[i-2][j] == EMPTY):
                        moves.append(((j,i),(j,i-2)))
        
                    # check if piece can move down
                    if i < 7 and state[i+1][j] == EMPTY:
                        moves.append(((j,i),(j,i+1)))
                    elif (i < 6 and state[i+1][j] != EMPTY 
                          and state[i+2][j] == EMPTY):
                        moves.append(((j,i),(j,i+2)))
                
                    # check if piece can move right
                    if j < 7 and state[i][j+1] == EMPTY:
                        moves.append(((j,i),(j+1,i)))
                    elif (j < 6 and state[i][j+1] != EMPTY 
                          and state[i][j+2] == EMPTY):
                        moves.append(((j,i),(j+2,i)))
                
                    # check if piece can move left
                    if j > 0 and state[i][j-1] == EMPTY:
                        moves.append(((j,i),(j-1,i)))
                    elif (j > 1 and state[i][j-1] != EMPTY 
                          and state[i][j-2] == EMPTY):
                        moves.append(((j,i),(j-2,i)))
        
        # return list of all available moves
        return moves 
    
    def make_move(self, state, action):
        '''
        
        Applies the given action to the given board state. 
        
        Parameters: 
            state ('list' of 'list' of 'char'): starting board configuration
            action ('tuple' of ('int','int')): given action in same format as 
            self.moves
        
        Returns
            'list' of 'list' of 'char': new board state after move is applied
            
        '''
        
        # retrieve coordinates from tuple
        x1 = action[0][0]
        y1 = action[0][1]
        x2 = action[1][0]
        y2 = action[1][1]
        
        player = state[y1][x1]
        
        new_state = deepcopy(state)
        
        # make the move 
        new_state[y1][x1] = EMPTY
        new_state[y2][x2] = player
        
        # apply any eliminations 
        new_state = self.elim(new_state,(x2,y2))
        
        # return the new board configuration
        return new_state
    
    def elim(self, state, last_move):
        
        '''
        
        Eliminates any pieces eliminated by the last played piece. 
        
        Assumes that there are no other eliminations to be made on board. 
        
        Parameters: 
            state ('list' of 'list' of 'char'): board configuration
            last_move ('tuple' of 'int'): coordinates of last moved piece 
        
        Returns: 
            'list' of 'list' of 'char': new board configuration
        
        '''
        
        # retrieve coordinates from tuple 
        x = last_move[0]
        y = last_move[1]
        
        player = state[y][x]
        
        new_state = deepcopy(state)
        
        # determine which player is opponent 
        if player == WHITE:
            opponent = BLACK
        else:
            opponent = WHITE
    
        # give player priority by eliminating any pieces it eliminates first 
        # check if an opponent is eliminated vertically 
        # apply any eliminations
        if y > 1 and new_state[y-1][x] == opponent and (new_state[y-2][x] == player 
                                                or new_state[y-2][x] == CORNER):
            new_state[y-1][x] = EMPTY
        if y < 6 and new_state[y+1][x] == opponent and (new_state[y+2][x] == player 
                                                or new_state[y+2][x] == CORNER):
            new_state[y+1][x] = EMPTY
    
        # check if an opponent is eliminated horizontally 
        # apply any eliminations
        if x > 1 and new_state[y][x-1] == opponent and (new_state[y][x-2] == player 
                                                or new_state[y][x-2] == CORNER):
            new_state[y][x-1] = EMPTY
        if x < 6 and new_state[y][x+1] == opponent and (new_state[y][x+2] == player 
                                                or new_state[y][x+2] == CORNER):
            new_state[y][x+1] = EMPTY
    
        # check if any remaining pieces will cause elimination for player piece 
        # check for vertical elimination
        # apply any eliminations
        if (y > 0 and y < 7 and new_state[y-1][x] == opponent 
            and (new_state[y+1][x] == opponent or new_state[y+1][x] == CORNER)):
            new_state[y][x] = EMPTY
            return new_state
        if (y > 0 and y < 7 and new_state[y+1][x] == opponent 
            and new_state[y-1][x] == CORNER):
            new_state[y][x] = EMPTY
            return new_state
    
        # check for horizontal elimination
        # apply any eliminations 
        if (x > 0 and x < 7 and new_state[y][x-1] == opponent 
            and (new_state[y][x+1] == opponent or new_state[y][x+1] == CORNER)):
            new_state[y][x] = EMPTY
            return new_state
        if (x > 0 and x < 7 and new_state[y][x+1] == opponent 
            and new_state[y][x-1] == CORNER):
            new_state[y][x] = EMPTY
            return new_state
        return new_state 
        
    
    def check_goal(self, state):
        '''
        
        Checks whether board state satisfies game goal. 
        
        Parameters:
            state ('list' of 'list' of 'char'): board configuration
        
        Returns: 
            'bool': True if board meets goal, False otherwise
            
        '''
        
        # count pieces in the goal 
        pieces = self.count_pieces(state)
        
        if self.goal[0] == DOESNT_MATTER and self.goal[1] == DOESNT_MATTER:
            # if there is no goal return true 
            return True
        elif self.goal[0] == DOESNT_MATTER:
            # check black pieces 
            if pieces[1] <= self.goal[1]:
                return True
        elif self.goal[1] == DOESNT_MATTER:
            # check white pieces 
            if pieces[0] <= self.goal[0]:
                return True 
        else: 
            if pieces[0] <= self.goal[0] and pieces[1] <= self.goal[1]:
                # check both totals 
                return True 
        return False  

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
    
    def __init__(self, game, state, parent=None, path=None,children=None,depth=0):
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
        # board configuration
        self.state = state
        # path from root to node 
        self.path = [] if path is None else path
        # children of node 
        self.children = []
        
        # generate children if available 
        if children is not None:
            for child in children:
                self.add_child(child)
        
        # depth in search tree 
        self.depth = depth
        
        # path cost from root to node
        self.g = depth
        
        # A* heuristic 
        self.h = self.game.h_factor*(round(self.game.count_pieces(self.state)[1]/3))
        
        # A* search score 
        self.f = self.g + self.h
        
        # whether node has been expanded in search tree 
        self.expanded = False 
    
    def __repr__(self):
        rep = ""
        for line in self.state:
            for char in line:
                rep.append(char + " ")
            rep.append('\n')
        return rep
        
    def __lt__(self, other):
        if hasattr(other, 'f'):
            return self.f.__lt__(other.f)
    
    def __eq__(self, other):
        if hasattr(other, 'f'):
            return self.f.__eq__(other.f)
    
    def __cmp__(self,other):
        if hasattr(other, 'f'):
            return self.f.__cmp__(other.f)
        
    def expand(self):
        '''
        
        Expands search tree at node to include all available child nodes. 
        
        '''
        
        # generate all moves available from current state 
        moves = self.game.moves(self.state, WHITE)
        
        # add child for each move 
        for move in moves:
            self.add_child(move)
        
        # sort children by f value 
        self.children = sorted(self.children)
        
        self.expanded = True 
    
    
    def add_child(self, move):
        '''
        
        Uses given move to generate appropriate child node and add it to 
        self.children. 
        
        Parameters:
            move ('tuple' of ('int','int)): move to be taken in same format 
            as returned in list in self.moves
        
        '''
        
        # make the move 
        new_state = self.game.make_move(self.state, move)
        
        # create node out of new game state 
        child = Node(game, new_state, self, self.path + [move], None,
                    self.depth+1)
        
        # append node to self.children
        self.children.append(child)
    
    
    def compare_nodes(self, other):
        return self.state == other.state


###########################################################################
###################### SEARCH FUNCTIONS ###################################
###########################################################################

def Astar(root):
    """
    Implements A* search algorithm using the heuristic that g(n) = n.depth
    and h(n) = n.pieces_remain. 
    
    Source: https://en.wikipedia.org/wiki/A*_search_algorithm#Description
    
    Parameters: 
        root ('Node'): root of the search tree
    
    Returns: 
        'Node': node containing the solution state 
    
    """
    
    # initialize open and closed lists 
    open_list = [root]
    closed_list = []
    
    # continue to loop while there are open nodes 
    while len(open_list) != 0:
        
        # chose node with highest f value from open list 
        # also remove it from list 
        current = open_list.pop(0)
        
        # if game has ended, return current node 
        if root.game.check_goal(current.state):
            return current
        
        # add current node to closed list 
        closed_list.append(current)
        closed_list = sorted(closed_list)
        
        # expand node if necessary 
        if current.expanded == False:
            current.expand()
         
        for child in current.children:
                        
            in_open = False
            in_closed = False
            
            # check if node is already closed 
            # break and continue if so 
            for node in closed_list:
                if child.state == node.state:
                    in_closed = True 
                    break
                
            if in_closed: 
                continue 
            
            # check if node has the same state as another
            # node in the open list
            # save node if otherwise 
            for node in open_list:
                if child.compare_nodes(node):
                    in_open = True
                    open_child = node
                    break
            
            # add child to open list if it is not already in it 
            if not in_open:
                open_list.append(child)
                open_list = sorted(open_list)
            else: 
                # if child has lower depth than its equivalent 
                # in the open list, replace the node in open list 
                if child.depth < open_child.depth:
                    open_child = child
        
    return None  

def DLS(root, depth):
    '''
    
    Implements a limited DFS algorithm 
    https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search
    
    Parameters: 
        root ('Node'): root tree node to search from 
        depth ('int'): depth to explore tree
    
    Returns: 
        'Node': solution node (if it exists, None otherwise)
        
    '''
    
    # if we are at the target depth for iterative_DLS
    # check if this is a solution and return
    # or else, expand the tree at this node
    # in anticipation of next stage of IDS 
    if depth == 0 and root.game.check_goal(root.state):
        return root 
    elif depth == 0:
        root.expand()
        return None 
    
    # if we are not yet at target depth
    # continue to iterate down and 
    # return only if solution is found 
    if depth > 0:
        for child in root.children:
            found = DLS(child, depth-1)
            if found != None:
                return found
    return None 

def iterative_DLS(root):
    '''
    
    Implements iterative DLS as shown in the following page
    https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search
    
    Parameters:
        root ('Node'): root node of search tree 
    
    Returns: 
        'Node': node containing solved game board (if it exists) 
    
    '''
    
    depth = 0
    while True: 
        found = DLS(root, depth)
        if found != None:
            return found
        depth += 1

def print_board(board):
    '''
    
    Prints the game board in the same form as project spec. 
    
    Included for debugging purposes. 
    
    Parameters:
        board ('list' of 'list' of 'char'): state of game board 
        
    '''
    for line in board:
        for char in line:
            print(char + " ",end="",flush=True)
        print()

###########################################################################
##################### BEGIN RUNNING CODE ##################################
###########################################################################

# initialize game board
board = []

# read in the board assuming that board is in correct format
# board is in the format of a list of list of characters
# with each character representing one space 
for i in range(0,8):
    line = ((sys.stdin.readline()).strip('\n')).replace(" ", "")
    board.append(list(line))

# create an instance of the game 
game = Game(board, (DOESNT_MATTER,0))

# read the chosen game mode
mode = (sys.stdin.readline()).strip('\n')

if mode != "Moves" and mode != "Massacre":
    # exit the program if the mode is invalid 
    print("Invalid mode entered. Exiting program...")
    quit()
elif mode == "Moves":
    # run the function to generate moves for both players 
    print(len(game.moves(game.initial,WHITE)))
    print(len(game.moves(game.initial,BLACK)))
elif mode == "Massacre":
    # create root of search tree
    root = Node(game, board)
    # generate solution for massacre 
    solution = Astar(root)
    # print path in required format 
    for step in solution.path:
        print("({:d}, {:d}) -> ({:d}, {:d})"
              .format(step[0][0],step[0][1],step[1][0],step[1][1]))
