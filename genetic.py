import random
import csv
from selfplay import self_play
from copy import deepcopy
'''
https://towardsdatascience.com/genetic-programming-for-ai-heuristic-optimization-9d7fdb115ee1
'''
POPULATION = 20
CYCLES = 12
MUTATION_CHANCE = 0.15

def generate_population(count):
    population = []
    with open('weights.csv') as file:
        contents = file.read()
        contents_list = contents.split(',')
        population.append([float(x) for x in contents_list])
    for _ in range(count-1):
        population.append([random.uniform(-10,10) for _ in range(12)])
    return population

def check_fitness(specimen):
    with open("weights.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(specimen)
    
    wins = 0
    
    for _ in range(50):
        _, winner = self_play('player2', 'tdlearn')
        if winner == 'B':
            wins += 1
    
    return wins

def breed(mother, father):
    child = []
    if mother != father:
        for i in range(12):
            if father[i] > mother[i]:
                child.append(random.uniform(mother[i]*0.95,father[i]*0.95))
            else:
                child.append(random.uniform(father[i]*0.95,mother[i]*0.95))
    
    return child

def mutate(specimen):
    new_specimen = list(deepcopy(specimen))
    for i in range(12):
        x = random.uniform(0,1)
        if x < MUTATION_CHANCE:
            new_specimen[i] += random.uniform(-0.5,0.5)
    return new_specimen

def evolve(pop):
    scores = {}
    for i in range(len(pop)):
        # blah
        score = check_fitness(pop[i])
        if tuple(pop[i]) not in scores.keys():
            scores[tuple(pop[i])] = score
    
    sorted_scores = sorted(scores,key=scores.get, reverse =True)
    
    best_weights = sorted_scores[0]
    best_score = scores[best_weights]
    
    retain = sorted_scores[:5]
    retain = retain + random.sample(sorted_scores[5:],1)
    random.shuffle(retain)
    
    num_children = len(pop) - len(retain)
    children = []
    for i in range(num_children):
        par = random.sample(retain, 2)
        child = breed(par[0],par[1])
        children.append(child)
        
    new_pop = children + retain
    
    mutated_pop = []
    for specimen in new_pop:
        mutated_pop.append(mutate(specimen))
    
    return mutated_pop, best_weights, best_score
    

population = generate_population(20)
best = 0
best_weights = []

for i in range(CYCLES):
    print("This is generation " + str(i) + ".")
    population, best_specimen, best_score = evolve(population)
    print("The best specimen was: " + str(best_specimen))
    print("This specimen scored " + str(best_score))
    if best_score > best:
        best_weights = best_specimen
        best = best_score

with open("weights.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(list(best_weights)) 

