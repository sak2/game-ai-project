from game2 import Game
FIRST_SHRINK = 152
SECOND_SHRINK = 216
WHITE = 'O'
BLACK = '@'
EMPTY = '-'
CORNER = 'X'
DEAD = 'D'

def convert_to_feature(game):
    num_white = 0
    num_black = 0
    
    white_free = 0
    black_free = 0
    wh_1 = 0; bl_1 = 0; wh_2 = 0; bl_2 = 0; wh_3 = 0; bl_3 = 0; wh_4 = 0; bl_4 = 0
    wh_wall = 0; bl_wall = 0; wh_corner = 0; bl_corner = 0
    wh_1_w = 0; wh_1_c = 0; wh_1_wh = 0; wh_1_bl = 0
    bl_1_w = 0; bl_1_c = 0; bl_1_wh = 0; bl_1_bl = 0
    wh_2_wc = 0; wh_2_whwh = 0; wh_2_blbl = 0; wh_2_wwh = 0; wh_2_wbl = 0; wh_2_whbl = 0
    bl_2_wc = 0; bl_2_whwh = 0; bl_2_blbl = 0; bl_2_wwh = 0; bl_2_wbl = 0; bl_2_blwh = 0
    wh_3_whwhwh = 0; wh_3_cwwh = 0; wh_3_cwbl = 0; wh_3_wblbl = 0;
    wh_3_wwhwh = 0; wh_3_wwhbl = 0; wh_3_whwhbl = 0; wh_3_whblbl = 0
    bl_3_blblbl = 0; bl_3_cwwh = 0; bl_3_cwbl = 0; bl_3_wblbl = 0; 
    bl_3_wwhwh = 0; bl_3_wwhbl = 0; bl_3_blblwh = 0; bl_3_blwhwh = 0
    wh_4_wcwhwh = 0; wh_4_wcwhbl = 0; wh_4_whwhwhwh = 0; wh_4_wwhwhwh = 0;
    wh_4_whwhwhbl = 0; wh_4_whwhblbl = 0
    wh_4_wwhwhbl = 0; wh_4_wwhblbl = 0
    bl_4_wcblbl = 0; bl_4_wcwhbl = 0; bl_4_blblblbl = 0; bl_4_wblblbl = 0
    bl_4_blblblwh = 0; bl_4_blblwhwh = 0
    bl_4_wblblwh = 0; bl_4_wblwhwh = 0
    
    
    
    turns_taken = game.turns

    if game.turns < FIRST_SHRINK:
        num_shrunk = 0
    elif FIRST_SHRINK <= game.turns < SECOND_SHRINK:
        num_shrunk = 1
    elif game.turns >= SECOND_SHRINK:
        num_shrunk = 2
    
    next_player = game.player
    
    if next_player == WHITE:
        white_turn = 1
        black_turn = 0
    elif next_player == BLACK:
        black_turn = 1
        white_turn = 0
        
    white_location = []
    black_location = []
    corner_location = []
    dead_cells = []
    
    for i in range(8):
        for j in range(8):
            
            if game.board[(i,j)] == WHITE:
                white_location.append(1)
                opp = 0
                own = 0
                wall = 0
                corner = 0
                num_white += 1
                if (i+1,j) in game.board.keys():
                    if game.board[(i+1,j)] == BLACK:
                        opp += 1
                    elif game.board[(i+1,j)] == 'X':
                        corner += 1
                    elif game.board[(i+1,j)] == WHITE:
                        own += 1
                    elif game.board[(i+1,j)] == DEAD:
                        wall += 1
                else:
                    wall += 1
                    
                if (i-1,j) in game.board.keys():
                    if game.board[(i-1,j)] == BLACK:
                        opp += 1
                    elif game.board[(i-1,j)] == 'X':
                        corner += 1
                    elif game.board[(i-1,j)] == WHITE:
                        own += 1
                    elif game.board[(i-1,j)] == DEAD:
                        wall += 1
                else:
                    wall += 1
                
                if (i,j+1) in game.board.keys():
                    if game.board[(i,j+1)] == BLACK:
                        opp += 1
                    elif game.board[(i,j+1)] == 'X':
                        corner += 1
                    elif game.board[(i,j+1)] == WHITE:
                        own += 1
                    elif game.board[(i,j+1)] == DEAD:
                        wall += 1
                else:
                    wall += 1
                
                if (i,j-1) in game.board.keys():
                    if game.board[(i,j-1)] == BLACK:
                        opp += 1
                    elif game.board[(i,j-1)] == 'X':
                        corner += 1
                    elif game.board[(i,j-1)] == WHITE:
                        own += 1
                    elif game.board[(i,j-1)] == DEAD:
                        wall += 1
                else:
                    wall += 1
                
                total = wall + corner + own + opp
                
                if wall:
                    wh_wall += 1
                if corner:
                    wh_corner += 1
                
                if total == 4:
                    wh_4 += 1
                    if wall and corner:
                        if own == 2:
                            wh_4_wcwhwh += 1
                        else:
                            wh_4_wcwhbl += 1
                    if own == 4:
                        wh_4_whwhwhwh += 1
                    if own == 3:
                        if wall:
                            wh_4_wwhwhwh += 1
                        else:
                            wh_4_whwhwhbl += 1
                    if own == 2:
                        if wall:
                            wh_4_wwhwhbl += 1
                        else:
                            wh_4_whwhblbl += 1
                    if own == 1:
                        wh_4_wwhblbl += 1
                if total == 3:
                    wh_3 += 1
                    if own == 3:
                        wh_3_whwhwh += 1
                    if own == 2:
                        if wall:
                            wh_3_wwhwh += 1
                        else:
                            wh_3_whwhbl += 1
                    if own == 1:
                        if corner:
                            wh_3_cwwh += 1
                        elif wall and not corner:
                            wh_3_wwhbl += 1
                        else:
                            wh_3_whblbl += 1
                    if own == 0:
                        if corner:
                            wh_3_cwbl += 1
                        if wall and not corner:
                            wh_3_wblbl += 1
                if total == 2:
                    wh_2 += 1
                    if own == 2:
                        wh_2_whwh += 1
                    if own == 1:
                        if wall:
                            wh_2_wwh += 1
                        else:
                            wh_2_whbl += 1
                    if own == 0:
                        if corner:
                            wh_2_wc += 1
                        elif wall and not corner:
                            wh_2_wbl += 1
                        else:
                            wh_2_blbl += 1
                if total == 1:
                    wh_1 += 1
                    if own:
                        wh_1_wh += 1
                    if opp:
                        wh_1_bl += 1
                    if wall:
                        wh_1_w += 1
                if total == 0:
                    white_free += 1         
            else:
                white_location.append(0)
            if game.board[(i,j)] == BLACK:
                black_location.append(1)
                opp = 0
                own = 0
                wall = 0
                corner = 0
                num_black += 1
                if (i+1,j) in game.board.keys():
                    if game.board[(i+1,j)] == WHITE:
                        opp += 1
                    elif game.board[(i+1,j)] == 'X':
                        corner += 1
                    elif game.board[(i+1,j)] == BLACK:
                        own += 1
                else:
                    wall += 1
                    
                if (i-1,j) in game.board.keys():
                    if game.board[(i-1,j)] == WHITE:
                        opp += 1
                    elif game.board[(i-1,j)] == 'X':
                        corner += 1
                    elif game.board[(i-1,j)] == BLACK:
                        own += 1
                else:
                    wall += 1
                
                if (i,j+1) in game.board.keys():
                    if game.board[(i,j+1)] == WHITE:
                        opp += 1
                    elif game.board[(i,j+1)] == 'X':
                        corner += 1
                    elif game.board[(i,j+1)] == BLACK:
                        own += 1
                else:
                    wall += 1
                
                if (i,j-1) in game.board.keys():
                    if game.board[(i,j-1)] == WHITE:
                        opp += 1
                    elif game.board[(i,j-1)] == 'X':
                        corner += 1
                    elif game.board[(i,j-1)] == BLACK:
                        own += 1
                else:
                    wall += 1
                
                total = wall + corner + own + opp
                
                if wall:
                    bl_wall += 1
                if corner:
                    bl_corner += 1
                
                if total == 4:
                    bl_4 += 1
                    if wall and corner:
                        if own == 2:
                            bl_4_wcblbl += 1
                        else:
                            bl_4_wcwhbl += 1
                    if own == 4:
                        bl_4_blblblbl += 1
                    if own == 3:
                        if wall:
                            bl_4_wblblbl += 1
                        else:
                            bl_4_blblblwh += 1
                    if own == 2:
                        if wall:
                            bl_4_wblblwh += 1
                        else:
                            bl_4_blblwhwh += 1
                    if own == 1:
                        bl_4_wblwhwh += 1
                if total == 3:
                    bl_3 += 1
                    if own == 3:
                        bl_3_blblbl += 1
                    if own == 2:
                        if wall:
                            bl_3_wblbl += 1
                        else:
                            bl_3_blblwh += 1
                    if own == 1:
                        if corner:
                            bl_3_cwbl += 1
                        elif wall and not corner:
                            bl_3_wwhbl += 1
                        else:
                            bl_3_blwhwh += 1
                    if own == 0:
                        if corner:
                            bl_3_cwwh += 1
                        if wall and not corner:
                            bl_3_wwhwh += 1
                if total == 2:
                    wh_2 += 1
                    if own == 2:
                        bl_2_blbl += 1
                    if own == 1:
                        if wall:
                            bl_2_wbl += 1
                        else:
                            bl_2_blwh += 1
                    if own == 0:
                        if corner:
                            bl_2_wc += 1
                        elif wall and not corner:
                            bl_2_wwh += 1
                        else:
                            bl_2_whwh += 1
                if total == 1:
                    bl_1 += 1
                    if own:
                        bl_1_bl += 1
                    if opp:
                        bl_1_wh += 1
                    if wall:
                        bl_1_w += 1
                if total == 0:
                    black_free += 1         
            else:
                black_location.append(0)
            
            if game.board[(i,j)] == CORNER:
                corner_location.append(1)
            else:
                corner_location.append(0)
            if game.board[(i,j)] == DEAD:
                dead_cells.append(1)
            else:
                dead_cells.append(0)
            
    return [turns_taken, white_turn, black_turn, num_shrunk, num_white, num_black,
            white_free, black_free, wh_1, bl_1, wh_2, bl_2, wh_3, bl_3, wh_4, bl_4, 
            wh_wall, bl_wall, wh_corner, bl_corner, wh_1_w, wh_1_c, wh_1_wh, wh_1_bl,
            bl_1_w, bl_1_c, bl_1_wh, bl_1_bl, wh_2_wc, wh_2_whwh, wh_2_blbl, 
            wh_2_wwh, wh_2_wbl, wh_2_whbl, bl_2_wc, bl_2_whwh, bl_2_blbl, bl_2_wwh,
            bl_2_wbl, bl_2_blwh, wh_3_whwhwh, wh_3_cwwh, wh_3_cwbl, wh_3_wblbl, 
            wh_3_wwhwh,wh_3_wwhbl, wh_3_whwhbl, wh_3_whblbl, bl_3_blblbl, bl_3_cwwh,
            bl_3_cwbl, bl_3_wblbl, bl_3_wwhwh, bl_3_wwhbl, bl_3_blblwh, bl_3_blwhwh,
            wh_4_wcwhwh, wh_4_wcwhbl, wh_4_whwhwhwh, wh_4_wwhwhwh, wh_4_whwhwhbl,
            wh_4_whwhblbl, wh_4_wwhwhbl, wh_4_wwhblbl, bl_4_wcblbl, bl_4_wcwhbl,
            bl_4_blblblbl, bl_4_wblblbl, bl_4_blblblwh, bl_4_blblwhwh, bl_4_wblblwh,
            bl_4_wblwhwh] + white_location + black_location + corner_location + dead_cells