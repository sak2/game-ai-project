import subprocess
from copy import deepcopy

def self_play(player1,player2):
    # run the program with the two players 
    stdoutdata = subprocess.check_output(['python','referee.py',player1,player2])
    
    # return the processed data as a tuple containing a list of dictionaries and the result
    return process_game(stdoutdata.decode("utf-8"))
    
def process_game(game):
    '''
    Returns the result of a game as a list of dictionaries
    and the result
    '''
    
    lines = game.splitlines()
    
    current_board_line = 0
    game_over = False
    
    states = []
    state = []
    
    for i in range(4,len(lines)): 
        
        if "game over!" in lines[i]:
            game_over = True
            continue
        
        if game_over:
            winner = lines[i][8]
            break
        
        if current_board_line == 8:
            current_board_line = 0
            states.append(state)
            state = deepcopy(state)
            state = []
            continue
        
        state.append(lines[i].split())
        current_board_line += 1
    
    dict_states = []
    for state in states:
        state_dict = {(x,y):state[y][x] for x in range(8) for y in range(8)}
        dict_states.append(state_dict)
    return dict_states, winner
    
    