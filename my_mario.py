import gym
import time as t
import ppaquette_gym_super_mario

import numpy as np
import matplotlib.pyplot as plt
from wrapper import action_space


env = gym.make('ppaquette/meta-SuperMarioBros-v0')

wra_act=action_space

#reduce actions
env=wra_act.mario_action(env)

#reduce pixel


env=wra_act.ProcessFrame84(env)


#environment reset

env.reset()



nothing = [0,0,0,0,0,0]
up      = [1,0,0,0,0,0]
left    = [0,1,0,0,0,0]
down    = [0,0,1,0,0,0]
right   = [0,0,0,1,0,0]
jump    = [0,0,0,0,1,0]
fire    = [0,0,0,0,0,1]

observe, reward, done, clear = env.step(right)

# np.set_printoptions(threshold=np.inf)
# print('observe.shape = \n', observe.shape)
# np.set_printoptions(threshold=1000)

env.close()

observe = np.reshape(observe, (84, 84))
observe_img = np.stack([observe, observe, observe], axis=2)
print(observe.shape)
print(observe_img.shape)
plt.imshow(observe_img)
plt.show()