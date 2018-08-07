import gym
import time as t
import ppaquette_gym_super_mario
import neural_net as nn
import tensorflow as tf
import math as m
import random as r

import numpy as np
import matplotlib.pyplot as plt
from wrapper import action_space


def GetQmax(state):
    Qmax = -m.inf
    Amax = 0
    for i in range(1, 15):
        action = np.array([[i]])
        Qvalue = dqn.predict(state, action)
        # action_Q = np.r_[action_Q, [i, Qvalue]]
        print('Action ', i, ' Qvalue : ', Qvalue)

        if Qmax < Qvalue: 
            Amax = i
            Qmax = Qvalue
    return Qmax

def ep_greedy(ep, state):
    action_Q = np.array([])
    Qmax = -m.inf
    Amax = 0
    for action in range(1, 15):
        Qvalue = dqn.predict(state, np.array([[action]]))
        if action == 1:
            action_Q = np.array([[action, Qvalue]])
        else:
            action_Q = np.r_[action_Q, np.array([[action, Qvalue]])]
        print('Action ', action, ' Qvalue : ', Qvalue)

        if Qmax < Qvalue: 
            Amax = action
            Qmax = Qvalue
    num = r.random()
    print('num :\n', num)
    if num < ep + (1-ep)/14:
        return Amax, Qmax
    else:
        action = r.randrange(1, 14)
        if action >= Amax:
            action += 1
        Qvalue = action_Q[action-1, 1]
        return action, Qvalue

    


env = gym.make('ppaquette/meta-SuperMarioBros-v0')

wra_act = action_space

env = wra_act.mario_action(env)

env = wra_act.ProcessFrame84(env)

observe = env.reset()
observe = np.reshape(observe, (1, 84, 84))
ObserveImg = np.stack([observe, observe, observe, observe], axis=3)

actions = {
    1  : [0,0,0,0,0,0], # Nothing
    2  : [1,0,0,0,0,0], # Up
    3  : [0,0,1,0,0,0], # Down
    4  : [0,1,0,0,0,0], # Left
    5  : [0,1,0,0,1,0], # Left + J
    6  : [0,1,0,0,0,1], # Left + F
    7  : [0,1,0,0,1,1], # Left + J + F
    8  : [0,0,0,1,0,0], # Right
    9  : [0,0,0,1,1,0], # Right + J
    10 : [0,0,0,1,0,1], # Rgiht + F
    11 : [0,0,0,1,1,1], # Right + J + F
    12 : [0,0,0,0,1,0], # J
    13 : [0,0,0,0,0,1], # F
    14 : [0,0,0,0,1,1]  # J + F
}

action_Q = []
Qmax = -m.inf
Amax = 0
gamma = 0.01
alpha = 0.001
buffer = [[]]
buffer_count = 0

g = tf.Graph()
with g.as_default():
    sess = tf.Session(graph=g)
    dqn = nn.Neural_Net(sess)
    sess.run(tf.global_variables_initializer())

    print('ObserveImg.shape :\n', ObserveImg.shape)

    action, Qvalue = ep_greedy(0.9, ObserveImg)
    print('Result : ', action, ' ', Qvalue)

    NewObserve, reward, done, clear = env.step(actions[action])
    
    NewObserve = np.reshape(NewObserve, (1, 84, 84, 1))
    print('NewObserve.shape : ', NewObserve.shape, 'ObserveImg.shape : ', ObserveImg.shape)
    NewObserveImg = np.concatenate([NewObserve, ObserveImg], axis=3)[:,:,:,0:4]

    Qmax = GetQmax(NewObserveImg)
    print('Qmax :\n', Qmax)
    Qvalue = Qvalue + alpha*(reward + gamma*Qmax - Qvalue)
    print('Qvalue :\n', Qvalue)

    if buffer_count == 0:
        buffer = [ObserveImg, action, Qvalue]
        buffer_count += 1
    elif buffer_count < 100:
        buffer.append([ObserveImg, action, Qvalue])
        buffer_count += 1
    else:
        batch = np.array(r.sample(buffer, 10))
        
        # Qvalue = dqn.train(state, np.array([[ActionBatch]]))
        buffer_count = 0

# np.set_printoptions(threshold=np.inf)
buffer = np.array(buffer)
print('buffer = \n', buffer)
# np.set_printoptions(threshold=1000)

env.close()



# print('observe.shape :\n', observe.shape)
# print('ObserveImg.shape :\n', ObserveImg.shape)
# print('reward :\n', reward)
# print('done \n', done)
# print('clear :\n', clear)
# observe = np.reshape(observe, (84, 84))
# observe_img = np.stack([observe, observe, observe], axis=2)
# plt.imshow(observe_img)
# plt.show()
