import tensorflow as tf
import numpy as np
import neural_net as nn

g = tf.Graph()
with g.as_default():
    Q_value = nn.neural_net()
    Q_target = reward + gamma * Q_value