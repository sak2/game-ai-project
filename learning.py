import random
import csv
from selfplay import self_play
from tdlearn import Player
from game2 import Game

'''
weights = [random.uniform(-1,1) for _ in range(11)]

with open("weights.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(weights)
'''
learner = Player('white')
wins = 0
last_10 = 0
beat_random = False
for i in range(1,1001):
    print("Game " + str(i) + " started.")
    game_states, winner = self_play('tdlearn','tdlearn')
    print("Game " + str(i) + " ended.")
    if winner == 'B':
        print("Black won the game.")
        wins += 1
        last_10 += 1
    if i % 10 == 0:
        print("Black won " + str(last_10) + " of its last 10 games.")
        last_10 = 0
    turns = 0
    game = []
    player = 'O'
    for i in range(len(game_states)):
        game.append(Game(game_states[i],player,turns))
        turns += 1
        if player == 'O':
            player = '@'
        else:
            player = 'O'
    learner.learn(game, winner)
    print("The current weights are: " + str(learner.weights))
print("TDLearn won " + str(wins) + " total games.")
