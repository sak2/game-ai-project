import numpy as np
from game2 import Game
from feature_conversion import convert_to_feature
from node import Node
import csv
from copy import deepcopy
import matplotlib.pyplot as plt
from selfplay import self_play


INF = float('inf')
FIRST_SHRINK = 152
SECOND_SHRINK = 216
WHITE = 'O'
BLACK = '@'

class Neural_Network(object):
    
    def __init__(self, model=None,input_size=328,output_size=2,hidden_size=656):
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        if model:
            with open(model) as f:
                reader = csv.reader(f)
                W1 = []
                W2 = []
                W3 = []
                b1 = []
                b2 = []
                b3 = []
                in_W1 = False
                in_W2 = False
                in_W3 = False
                in_b1 = False
                in_b2 = False
                in_b3 = False
                for row in reader:
                    if row[0] == 'W1':
                        in_W1 = True
                        in_W2 = False
                        in_W3 = False
                        in_b1 = False
                        in_b2 = False
                        in_b3 = False
                        continue
                    if row[0] == 'W2':
                        in_W1 = False
                        in_W2 = True
                        in_W3 = False
                        in_b1 = False
                        in_b2 = False
                        in_b3 = False
                        continue
                    if row[0] == 'W3':
                        in_W1 = False
                        in_W2 = False
                        in_W3 = True
                        in_b1 = False
                        in_b2 = False
                        in_b3 = False
                        continue
                    if row[0] == 'b1':
                        in_W1 = False
                        in_W2 = False
                        in_W3 = False
                        in_b1 = True
                        in_b2 = False
                        in_b3 = False
                        continue
                    if row[0] == 'b2':
                        in_W1 = False
                        in_W2 = False
                        in_W3 = False
                        in_b1 = False
                        in_b2 = True
                        in_b3 = False
                        continue
                    if row[0] == 'b3':
                        in_W1 = False
                        in_W2 = False
                        in_W3 = False
                        in_b1 = False
                        in_b2 = False
                        in_b3 = True
                        continue
                    
                    row = list(filter(None,row))
                    if in_W1:
                        W1.append(row)
                    if in_W2:
                        W2.append(row)
                    if in_W3:
                        W3.append(row)
                    if in_b1:
                        b1.append(row)
                    if in_b2:
                        b2.append(row)
                    if in_b3:
                        b3.append(row)
                self.model = {}
                self.model['W1'] = np.asarray(W1,dtype=float)
                self.model['W2'] = np.asarray(W2,dtype=float)
                self.model['W3'] = np.asarray(W3,dtype=float)
                self.model['b1'] = np.asarray(b1,dtype=float)
                self.model['b2'] = np.asarray(b2,dtype=float)
                self.model['b3'] = np.asarray(b3,dtype=float)
        if not model:
            # for testing purposes seed the random number generator
            # this ensures same results each time 
            self.model = {}
            
            # generate random input to hidden layer weights
            self.model['W1'] = np.random.randn(self.input_size,self.hidden_size) / np.sqrt(self.input_size)
            # generate random hidden to output layer weights 
            self.model['W2'] = np.random.randn(self.hidden_size,self.hidden_size) / np.sqrt(self.hidden_size)
            self.model['W3'] = np.random.randn(self.hidden_size,self.output_size) / np.sqrt(self.hidden_size)
            
            # initialise hidden layer bias 
            self.model['b1'] = np.zeros((1,self.hidden_size))
            self.model['b2'] = np.zeros((1,self.hidden_size))
            
            # initialise output layer bias  
            self.model['b3'] = np.zeros((1,self.output_size))
        
        # constant to prevent division by zero in Adam optimisation
        self.eps = 0.000000001
        
        # initial learning rate 
        self.alpha = 0.001
        
        # discount factor for TDLeaf(lambda)
        self.l = 0.6
        
        # regularisation factor 
        self.reg = 0.01
        
        # determine whether to perform dropout regularisation 
        # set to false on default as only used in training 
        self.do_dropout = False
        
        # set probability of neuron dropout 
        self.p_dropout = 0.1
    
    def forward_prop(self, x):
        '''
        Forward propagate input vector through neural net
        '''
        z1 = np.dot(x, self.model['W1'])  + self.model['b1']
        a1 = relu(z1)
        u1 = np.random.binomial(1,0,size=np.shape(a1))/self.p_dropout
        if self.do_dropout:
            u1 = np.random.binomial(1,self.p_dropout,size=np.shape(a1))/self.p_dropout
            a1 *= u1
        z2 = np.dot(a1,self.model['W2']) + self.model['b2']
        a2 = relu(z2)
        u2 = np.random.binomial(1,0,size=np.shape(a2))/self.p_dropout
        if self.do_dropout:
            u2 = np.random.binomial(1,self.p_dropout,size=np.shape(a2))/self.p_dropout
            a2 *= u2
        z3 = np.dot(a2,self.model['W3']) + self.model['b3']
        a3 = softmax(z3)
        return {'x':x,'z1':z1,'a1':a1,'u1':u1,'z1':z2,'a2':a2, 'u2':u2,'z3':z3,'a3':a3}
    
    def backward_prop(self, cache, y):
        '''
        Calculate derivatives of loss function with respect to weights 
        by backpropagation algorithm 
        '''
        W1, W2, W3 = self.model['W1'], self.model['W2'], self.model['W3']
        b1, b2, b3 = self.model['b1'], self.model['b2'], self.model['b3']
    
        # Load forward propagation results
        x, a1, a2, a3, u1, u2 = cache['x'],cache['a1'],cache['a2'], cache['a3'],cache['u1'], cache['u2']
        
        m = np.shape(y)[0]
        
        # get number of samples
        if m == 1:
            x = [x]
        
        # calculate the error 
        delta3 = loss_derivative(y,a3)
        dW3 = (1/m) * np.dot(np.transpose(a2),delta3)
        db3 = (1/m) * np.sum(delta3,axis=0,keepdims=True)
        
        da2 = np.dot(delta3,np.transpose(W3))
        if self.do_dropout:
            da2 *= u2
        
        delta2 = da2 *relu_derivative(a2)
        dW2 = (1/m) * np.dot(np.transpose(a1),delta2)
        db2 = (1/m) * np.sum(delta2,axis=0,keepdims=True)
        
        da1 = np.dot(delta2,np.transpose(W2))
        if self.do_dropout:
            da1 *= u1
        delta1 = da1 * relu_derivative(a1)
        dW1 = (1/m) * np.dot(np.transpose(x),delta1)
        db1 = (1/m) * np.sum(delta1,axis=0)
    
        
        return {'W1':dW1,'W2':dW2, 'W3':dW3, 'b1':db1,'b2':db2,'b3':db3}
    
    def update_parameters(self,grads):
        '''
        Apply update in direction of supplied gradients
        '''
        for key in grads:
            self.model[key] -= self.alpha*grads[key]
        
        return
    
    def predict(self, x):
        '''
        Provide index of column predicted to 
        receive greatest reward from given state x
        '''
        c = self.forward_prop(x)
        y_hat = np.argmax(c['a3'], axis=1)
        return y_hat
    
    def accuracy_score(self,prediction,actual):
        '''
        Calculate accuracy of predictions 
        '''
        m = len(prediction)
        correct = 0
        for i in range(m):
            if prediction[i] == actual[i]:
                correct += 1
        return float(correct)/m
    '''
    def train(self,X,Y,epochs = 20000, print_loss=False):
        
        beta1 = 0.9
        beta2 = 0.999
        
        M = {k:np.zeros_like(v) for k, v in self.model.items()}
        R = {k:np.zeros_like(v) for k, v in self.model.items()}
        
        for i in range(epochs):
            
            XY = list(zip(X,Y))
            random.shuffle(XY)
            X = [e[0] for e in XY]
            Y = [e[1] for e in XY]
            cache = self.forward_prop(X)
            
            grads = self.backward_prop(cache,Y)
            for k in grads:
                
                M[k] = beta1*M[k] + (1 - beta1)*grads[k]
                R[k] = beta2*R[k] + (1 - beta2)*grads[k]**2
                
                m_k_hat = M[k] / (1 - beta1**(i))
                r_k_hat = R[k] / (1 - beta2**(i))
                
                self.model[k] -= self.alpha*m_k_hat/(np.sqrt(r_k_hat) + self.eps)
            
            if print_loss and i % 100 == 0:
                a2 = cache['a2']
                print('Loss after iteration',i,':',softmax_loss(Y, a2))
                y_hat = self.predict(X)
                y_true = np.argmax(Y, axis=1)
                print('Accuracy after iteration',i,':',self.accuracy_score(y_hat,y_true)*100,'%')
                self.save_model()
        return
       '''
   
    def play_train(self,episodes = 5000, random_test = True, regularise = False,adam=False):
        '''
        Train neural network via repeated self-play
        Uses TDLeaf(Lambda) algorithm, Adam optimisation and dropout regularisation
        '''
        # turn on dropout regularisation for training
        if regularise:
            self.do_dropout = True 
        
        # set Adam optimisation constants 
        beta1 = 0.9
        beta2 = 0.999
        
        # initialise momentum and RMS matrices for Adam optimisation
        M = {k:np.zeros_like(v) for k, v in self.model.items()}
        R = {k:np.zeros_like(v) for k, v in self.model.items()}
        
        averages = []
        average_loss = []
        
        # play given number of episodes 
        for episode in range(1,episodes + 1):
            
            # initialise eligibility traces to zero at start of every episode
            trace1 = np.zeros((self.input_size,self.hidden_size))
            trace2 = np.zeros((self.hidden_size,self.hidden_size))
            trace3 = np.zeros((self.hidden_size,self.output_size))
            trace4 = np.zeros((1,self.hidden_size))
            trace5 = np.zeros((1,self.hidden_size))
            trace6 = np.zeros((1,self.output_size))
            traces = {'W1':trace1,'W2':trace2, 'W3':trace3, 'b1':trace4,'b2':trace5, 'b3':trace6}
            
            # initialise the game 
            game = Game()
            
            x = deepcopy(game)
            
            game_step = 0
            
            reward = None
            print("Game",str(episode),"started.")
            
            td = []
            losses = []
            
            # continue to loop while game is non-terminal 
            while not game.reward():
                
                # generate list of legal moves 
                moves = game.moves()
                
                if moves == []:
                    # if no moves available, just forfeit this turn
                    game.make_move(None)
                elif len(moves) == 1:
                    # if only one move available, just make it 
                    game.make_move(moves[0])
                else:
                    if game.player == WHITE:
                        index = 0
                    else:
                        index = 1
                    
                    scores = []
                    for move in moves:
                        new_game = deepcopy(game)
                        new_game.make_move(move)
                        scores.append((move,self.forward_prop(convert_to_feature(new_game))['a3'][0][index]))
                    
                    best = max(scores,key=lambda item:item[1])[0]
                    game.make_move(best)
                
                
                x_next = deepcopy(game)
                V_next = self.forward_prop(convert_to_feature(x_next))['a3']
                # forward propogate the current x leaf node 
                cache = self.forward_prop(convert_to_feature(x))
                d_t = np.sum(V_next - cache['a3'])
                
                td.append(d_t)
                
                # retrieve gradients 
                grads = self.backward_prop(cache, np.asarray(V_next))
                
                update = {}
                for key in grads:
                    # calculate eligibility traces
                    traces[key] = self.l*traces[key] + grads[key]
                    traces[key] = traces[key]
                    
                    # multiply by temporal difference 
                    update[key] = traces[key]*d_t
                    
                    if adam:
                        # apply Adam optimisation calculations
                        M[key] = beta1 * M[key] + (1 - beta1) * update[key]
                        R[key] = beta2 * R[key] + (1 - beta2) * update[key]**2
                        m_k_hat = M[key]/(1-beta1**(episode))
                        r_k_hat = R[key]/(1 - beta2**(episode))
                        update[key] = m_k_hat/(np.sqrt(r_k_hat) + self.eps)
                    
                # apply gradient update to weights 
                self.update_parameters(update)
                
                losses.append(softmax_loss(cache['a3'], np.asarray(V_next)))
                # set the next state to be the current state 
                x = x_next
                game_step += 1
                        
            cache = self.forward_prop(convert_to_feature(x))
            reward = game.reward()
            d_t = np.sum([reward] - cache['a3'])
            td.append(d_t)
            
            grads = self.backward_prop(cache, [reward])
            
            update = {}
            
            for key in grads:
                # calculate eligibility traces
                traces[key] = self.l*traces[key] + grads[key]
                traces[key] = traces[key]
                # multiply by temporal difference 
                update[key] = traces[key]*d_t
                
                if adam:
                    # apply Adam optimisation calculations
                    M[key] = beta1 * M[key] + (1 - beta1) * update[key]
                    R[key] = beta2 * R[key] + (1 - beta2) * update[key]**2
                    m_k_hat = M[key]/(1-beta1**(episode))
                    r_k_hat = R[key]/(1 - beta2**(episode))
                    update[key] = m_k_hat/(np.sqrt(r_k_hat) + self.eps)
            
            self.update_parameters(update)
            
                    
            print("Game",str(episode),"ended.")
            print("There was",game_step,"turns.")
            print("Average temporal difference was",str(np.sum(td)/game_step))
            print("Average softmax loss was",str(np.sum(losses)/game_step))
            averages.append(np.sum(td)/game_step)
            average_loss.append(np.sum(losses)/game_step)
            if reward == [1,0]:
                print("White wins")
            if reward == [0.5,0.5]:
                print("Draw")
            if reward == [0,1]:
                print("Black wins")
            
            # save the model generated by the game 
            self.save_model()    
            
            if episode % 100 == 0 and random_test:
                p1_wins = 0
                p1_draw = 0
                p2_wins = 0
                p2_draw = 0
                for _ in range(25):
                    _, winner = self_play('player1','neural_play')
                    if winner == 'B':
                        p2_wins += 1
                    if winner not in 'WB':
                        p2_draw += 1
                    _, winner = self_play('neural_play','player1')
                    if winner == 'W':
                        p1_wins += 1
                    if winner not in 'WB':
                        p1_draw += 1
                print("As player 1, neural_play won",str(p1_wins),"games and drew",str(p1_draw))
                print("As player 2, neural_play won",str(p2_wins),"games and drew",str(p2_draw))
        plt.plot(average_loss)
        plt.show()
                
        
    def save_model(self):
        '''
        Save the model contained in neural net to 'neural.csv'
        '''
        with open("neural.csv","w",newline="") as f:
            w = csv.writer(f)
            w.writerow(['W1'])
            for row in self.model['W1']:
                w.writerow(row)
            w.writerow(['W2'])
            for row in self.model['W2']:
                w.writerow(row)
            w.writerow(['W3'])
            for row in self.model['W3']:
                w.writerow(row)
            w.writerow(['b1'])
            for row in self.model['b1']:
                w.writerow(row)
            w.writerow(['b2'])
            for row in self.model['b2']:
                w.writerow(row)
            w.writerow(['b3'])
            for row in self.model['b3']:
                w.writerow(row)
        
def relu(z):
    x = deepcopy(z)
    x[x<0] = 0
    return x

def relu_derivative(x):
    z = deepcopy(x)
    z[z<=0] = 0
    z[z>0] = 1
    return z 
    

def softmax(z):
    exp_scores = np.exp(z - np.max(z))
    if len(np.shape(exp_scores)) != 1:
        return exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
    else:
        return exp_scores/np.sum(exp_scores)

def softmax_loss(y,y_hat):
    '''
    Softmax cross entropy loss
    '''
    # clipping value
    minval = 0.000000000001
    # number of samples
    m = np.shape(y)[0]
    # loss formula
    loss = -1/m * np.sum(y*np.log(np.clip(y_hat,a_min=minval,a_max=None)))
    return loss

def loss_derivative(y,y_hat):
    '''
    Softmax cross entropy loss derivative
    '''
    return (y_hat - y)

def tanh_derivative(x):
    return (1 - np.power(x,2)) 

def minimax(root, net, index, depth=4,cutoff_test=None):
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
    def max_value(root, net, index, alpha, beta, depth):
        if cutoff_test(depth):
            features = convert_to_feature(root.game)
            result = root.game.check_goal()
            if result[0]:
                return root.game.reward()[index], root
            else:
                return net.forward_prop(features)['a3'][0][index], root 
            
        
        v = -INF
        
        if not root.expanded:
            root.expand()
        node = None
        for child in root.children:
            new_v, new_node = min_value(child,net,index, alpha,beta,depth+1)
            if new_v > v:
                node = new_node
                v = new_v
            if v >= beta:
                return v, node
            alpha = max(alpha,v)
        
        return v, node
    
    
    def min_value(root, net,index, alpha, beta, depth):
        if cutoff_test(depth):
            features = convert_to_feature(root.game)
            result = root.game.check_goal()
            if result[0]:
                return root.game.reward()[index], root
            else:
                return net.forward_prop(features)['a3'][0][index], root 
        v = INF
        node = None
        if not root.expanded:
            root.expand()
        for child in root.children:
            new_v, new_node = max_value(child,net,index,alpha,beta,depth+1)
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
        v, node = min_value(child, net, index, best_score, beta, 1)
        if v > best_score:
            best_score = v 
            best_action = child 
            best_node = node
    return best_action, v, best_node

net = Neural_Network('neural.csv')
net.play_train(5000, True, True, True)


