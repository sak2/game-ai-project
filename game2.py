'''

COMP30024 Project Game module 
Semester 1 2018
Tamara Hogan (682452) & Saleh Ahmed Khan (798838)

'''

from copy import deepcopy
WHITE = 'O'
BLACK = '@'
EMPTY = '-'
CORNER = 'X'
DEAD = 'D'

# empty game board 
INITIAL = [['X', '-', '-', '-', '-', '-', '-', 'X'],
           ['-', '-', '-', '-', '-', '-', '-', '-'],
           ['-', '-', '-', '-', '-', '-', '-', '-'],
           ['-', '-', '-', '-', '-', '-', '-', '-'],
           ['-', '-', '-', '-', '-', '-', '-', '-'],
           ['-', '-', '-', '-', '-', '-', '-', '-'],
           ['-', '-', '-', '-', '-', '-', '-', '-'],
           ['X', '-', '-', '-', '-', '-', '-', 'X']]

FIRST_SHRINK = 152
SECOND_SHRINK = 216

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
        board ('dict'): representation of the board as 2D array 
        player ('char'): the current active player (waiting to move)
        turns ('int'): the number of turns that have occurred 
        shrunk ('int'): how many board shrinks have occurred (max 2)
    
    '''
    def __init__(self, initial=None,player=WHITE,turns=0):
        
        '''
        
        Parameters: 
            initial ('dict'):  initial board configuration
            player ('char'): the current player 
            turns ('int'): current elapsed turns 
            
        '''
        
        # initialize the empty board
        self.board = {}
        
        # if no board is presented create the empty board 
        # otherwise use the presented board 
        if initial == None:
            for i in range(len(INITIAL)):
                for j in range(len(INITIAL)):
                    self.board[(i,j)] = INITIAL[j][i]
        else:
            self.board = deepcopy(initial)
        
        # the current player
        self.player = player
        
        # elapsed number of turns
        self.turns = turns
        
        # number of times the board has been shrunk 
        self.shrunk = 0
        
        
    def count_pieces(self):
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
        for coord in self.board:
            if self.board[coord] == WHITE:
                white += 1
            if self.board[coord] == BLACK:
                black += 1

        # return totals 
        return (white,black) 
    
    def moves(self):
        '''
        Returns a list of all valid moves for the current player. 
        '''
        moves = []
        
        # determine the phase we are in based on turn number
        if self.turns < 24:
            moves = self.place_moves()
        else: 
            moves = self.move_moves()
        return moves
    
    def place_moves(self):
        
        '''
        Returns a list of all valid placements for the current player
        '''
        
        moves = []
        
        # if the player is white use the white playing zone 
        if self.player == WHITE: 
            for i in range(0,6):
                for j in range(0,8):
                    if self.board[(j,i)] == EMPTY:
                        moves.append((j,i))
        
        # if the player is black use the black playing zone 
        if self.player == BLACK:
            for i in range(2,8):
                for j in range(0,8):
                    if self.board[(j,i)] == EMPTY:
                        moves.append((j,i))
                    
        return moves 
    
    def move_moves(self):
        '''
        
        Generates all available moves for the current player on the current 
        board configuration. 
        
        Returns: 
            'list' of (('int,'int'),('int','int')): all available moves in the 
            format of the initial position of the piece followed by the new 
            position of the piece 
        
        '''
        
        # initialize empty list of moves 
        moves = []
        
        # current amount of board shrink
        s = self.shrunk
        
        # iterate over the board 
        for i in range(0,8):
            for j in range(0,8):
                
                # if a space contains given player, we check for moves 
                if self.board[(j,i)] == self.player:
                    
                    # check if piece can move up 
                    if i > 0 + s and self.board[(j,i-1)] == EMPTY:
                        moves.append(((j,i),(j,i-1)))
                    elif (i > 1 + s and self.board[(j,i-1)] != EMPTY 
                          and self.board[(j,i-2)] == EMPTY):
                        moves.append(((j,i),(j,i-2)))
        
                    # check if piece can move down
                    if i < 7 - s and self.board[(j,i+1)] == EMPTY:
                        moves.append(((j,i),(j,i+1)))
                    elif (i < 6 - s and self.board[(j,i+1)] != EMPTY 
                          and self.board[(j,i+2)] == EMPTY):
                        moves.append(((j,i),(j,i+2)))
                
                    # check if piece can move right
                    if j < 7-s and self.board[(j+1,i)] == EMPTY:
                        moves.append(((j,i),(j+1,i)))
                    elif (j < 6-s and self.board[(j+1,i)] != EMPTY 
                          and self.board[(j+2,i)] == EMPTY):
                        moves.append(((j,i),(j+2,i)))
                
                    # check if piece can move left
                    if j > 0+s and self.board[(j-1,i)] == EMPTY:
                        moves.append(((j,i),(j-1,i)))
                    elif (j > 1+s and self.board[(j-1,i)] != EMPTY 
                          and self.board[(j-2,i)]== EMPTY):
                        moves.append(((j,i),(j-2,i)))
        
        # return list of all available moves
        return moves 
    
    def make_move(self, action=None):
        '''
        
        Applies the given action to the given board state. 
        
        Assumes that the given action is valid 
        
        Parameters: 
            action: given action in format given in project spec 
            
        '''
        
        if action == None:
            # if the action is forfeited just move on 
            self.update_turns()
            return
        elif isinstance(action[0], int):
            # placing phase move 
            
            # place the piece 
            self.board[action] = self.player
            
            # eliminate any pieces
            self.elim(action)
            
            # update game state 
            self.update_turns()
            
            return
        else:         
            # make the move 
            self.board[action[0]] = EMPTY
            self.board[action[1]] = self.player
        
            # apply any eliminations 
            self.elim(action[1])
            
            # update the game state
            self.update_turns()
            
            return 
    
    def elim(self, last_move):
        
        '''
        
        Eliminates any pieces eliminated by the last played piece. 
        
        Assumes that there are no other eliminations to be made on board. 
        
        Parameters: 
            last_move ('tuple' of 'int'): coordinates of last moved piece 
        
        '''

        # check what character is at the point of last move        
        player = self.board[last_move]
        
        # retrieve the co-ords of the last move
        x = last_move[0]
        y = last_move[1]
        
        # determine which player is opponent 
        # apply board shrink corner elimination if necessary 
        if player == WHITE:
            opponent = BLACK
        elif player == BLACK:
            opponent = WHITE
        elif player == 'X': 
            s = self.shrunk
            # board shrink eliminations 
            if x < 6-s and (self.board[(x+2,y)] not in 'D-' 
                          and self.board[(x+1,y)] != self.board[(x+2,y)]):
                self.board[(x+1,y)] = EMPTY
            if x > 2+s and (self.board[(x-2,y)] not in 'D-' 
                          and self.board[(x-1,y)] != self.board[(x-2,y)]):
                self.board[(x-1,y)] = EMPTY
            if y < 6-s and (self.board[(x,y+2)] not in 'D-' 
                          and self.board[(x,y+1)] != self.board[(x,y+2)]):
                self.board[(x,y+1)] = EMPTY
            if y > 2+s and (self.board[(x,y-2)] not in 'D-' 
                          and self.board[(x,y-1)] != self.board[(x,y-2)]):
                self.board[(x,y-1)] = EMPTY
            return
                
    
        # give player priority by eliminating any pieces it eliminates first 
        # check if an opponent is eliminated vertically 
        # apply any eliminations
        if y > 1 and self.board[(x,y-1)] == opponent and (self.board[(x,y-2)] == player 
                                                or self.board[(x,y-2)] == CORNER):
            self.board[(x,y-1)] = EMPTY
        if y < 6 and self.board[(x,y+1)] == opponent and (self.board[(x,y+2)] == player 
                                                or self.board[(x,y+2)] == CORNER):
            self.board[(x,y+1)] = EMPTY
    
        # check if an opponent is eliminated horizontally 
        # apply any eliminations
        if x > 1 and self.board[(x-1,y)] == opponent and (self.board[(x-2,y)] == player 
                                                or self.board[(x-2,y)] == CORNER):
            self.board[(x-1,y)] = EMPTY
        if x < 6 and self.board[(x+1,y)] == opponent and (self.board[(x+2,y)]== player 
                                                or self.board[(x+2,y)] == CORNER):
            self.board[(x+1,y)] = EMPTY
    
        # check if any remaining pieces will cause elimination for player piece 
        # check for vertical elimination
        # apply any eliminations
        if (y > 0 and y < 7 and self.board[(x,y-1)] == opponent 
            and (self.board[(x,y+1)] == opponent or self.board[(x,y+1)] == CORNER)):
            self.board[(x,y)] = EMPTY
            return
        if (y > 0 and y < 7 and self.board[(x,y+1)] == opponent 
            and self.board[(x,y-1)] == CORNER):
            self.board[(x,y)] = EMPTY
            return 
    
        # check for horizontal elimination
        # apply any eliminations 
        if (x > 0 and x < 7 and self.board[(x-1,y)] == opponent 
            and (self.board[(x+1,y)]== opponent or self.board[(x+1,y)] == CORNER)):
            self.board[(x,y)]= EMPTY
            return 
        if (x > 0 and x < 7 and self.board[(x+1,y)] == opponent 
            and self.board[(x-1,y)] == CORNER):
            self.board[(x,y)] = EMPTY
            return
        return 
        
    
    def check_goal(self):
        '''
        
        Checks whether board state satisfies game goal. 
        
        Returns: 
            ('bool','char'): True if board meets goal, False otherwise, with winner
            
        '''
        
        # count pieces in the goal 
        pieces = self.count_pieces()
        
        # determine if game is over 
        if pieces[0] < 2 and pieces[1] < 2:
            return (True, 0)
        if pieces[0] < 2 and pieces[1] >= 2:
            return (True, BLACK)
        if pieces[0] >= 2 and pieces[1] < 2:
            return (True, WHITE)
        
        # return if game is not over 
        return (False, None)
    
    def shrink_board(self): 
        
        '''
        Shrink the board according to the number of shrinks already completed
        '''
        
        s = self.shrunk
        
        # loop through the board, excluding already dead cells 
        for i in range(0+s,8-s):
            for j in range(0+s,8-s):
                if i == s or j == s or i == 7-s or j == 7-s:
                    # make the outer edge dead 
                    self.board[(i,j)] = DEAD
                    
        # create new corners in anti-clockwise order
        for corner in [(s+1,s+1),(s+1,6-s),(6-s,6-s),(6-s,s+1)]:
            self.board[corner] = CORNER
            # eliminate pieces eliminated by corner
            self.elim(corner)
            
        # increment shrink counter
        self.shrunk += 1
        return
    
    def update_turns(self):
        
        '''
        Provide any game state update information
        '''
        
        # increase the turn counter 
        self.turns += 1
        
        # shrink the board if necessary 
        if self.turns == FIRST_SHRINK or self.turns == SECOND_SHRINK:
            self.shrink_board()
        
        # alternate the player 
        if self.player == WHITE:
            self.player = BLACK
        elif self.player == BLACK:
            self.player = WHITE
            
        return
    
    def print_board(self):
        '''
    
        Prints the game board in the same form as project spec. 
    
        Included for debugging purposes. 
        
        '''
        
        # loop through all coordinates of the board and print them 
        for i in range(0,8):
            for j in range(0,8):
                if self.board[(j,i)] == DEAD:
                    print ("  ",end="",flush=True)
                else:
                    print(self.board[(j,i)] + " ",end="",flush=True)
            print()


