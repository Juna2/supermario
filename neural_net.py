import tensorflow as tf
import numpy as np

class Neural_Net:
    def __init__(self, session):
        self.session = session
        self.error = 0
        self.learning_rate = 0.01
        self.neural_net()

    def neural_net(self):
        self.tf_x = tf.placeholder(tf.float32, shape=[1, 84, 84, 4], name='tf_x')
        self.tf_x2 = tf.placeholder(tf.float32, shape=[1, 1], name='tf_x2')
        self.tf_y = tf.placeholder(tf.float32, shape=[1, 1], name='tf_y')

        with tf.variable_scope('conv_1'):
            w1 = tf.get_variable(name='_weights', shape=[3, 3, 4, 32])
            b1 = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[32]))
            conv1 = tf.nn.conv2d(input=self.tf_x, filter=w1, 
                                    strides=[1, 1, 1, 1], padding='VALID')
            conv1 = tf.nn.bias_add(conv1, b1, name='net_preactivation')
            conv1 = tf.nn.relu(conv1, name='activation')
        '''conv1 shape : [1, 82, 82, 32]'''
        print('conv1 :\n', conv1)

        conv1_pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='conv1_pool')
        '''conv1_pool shape : [1, 41, 41, 32]'''                                
        print('conv1_pool :\n', conv1_pool)

        with tf.variable_scope('conv_2'):
            w2 = tf.get_variable(name='_weights', shape=[3, 3, 32, 64])
            b2 = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[64]))
            conv2 = tf.nn.conv2d(input=conv1_pool, filter=w2,
                                    strides=[1, 1, 1, 1], padding='VALID')
            conv2 = tf.nn.bias_add(conv2, b2, name='net_preactivation')
            conv2 = tf.nn.relu(conv2, name='activation')
        '''conv2 shape : [1, 39, 39, 64]'''
        print('conv2 :\n', conv2)

        conv2_pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='conv2_pool')
        '''conv2_pool shape : [1, 20, 20, 64]'''
        print('conv2_pool :\n', conv2_pool)

        with tf.variable_scope('conv_3'):
            w3 = tf.get_variable(name='_weights', shape=[5, 5, 64, 128])
            b3 = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[128]))
            conv3 = tf.nn.conv2d(input=conv2_pool, filter=w3,
                                    strides=[1, 1, 1, 1], padding='VALID')
            conv3 = tf.nn.bias_add(conv3, b3, name='net_preactivation')
            conv3 = tf.nn.relu(conv3, name='activation')
        '''conv3 shape : [1, 16, 16, 128]'''
        print('conv3 :\n', conv3)

        conv3_pool = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='conv3_pool')
        '''conv3_pool shape : [1, 8, 8, 128]'''
        print('conv3_pool :\n', conv3_pool)

        conv3_pool_units = np.prod(conv3_pool.get_shape().as_list()[1:])
        print('conv3_pool_units :\n', conv3_pool_units)
        conv3_pool_reshape = tf.reshape(conv3_pool, shape=(-1, conv3_pool_units))
        print('conv3_pool_reshape :\n', conv3_pool_reshape)
        '''conv3_pool_reshape shape : [1, 8192]'''

        Conv3WithTf_x2 = tf.concat([conv3_pool_reshape, self.tf_x2], 1)

        with tf.variable_scope('fcl_1'):
            w4 = tf.get_variable(name='_weights', shape=[8193, 8193])
            b4 = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[8193]))
            fcl1 = tf.matmul(Conv3WithTf_x2, w4)
            fcl1 = tf.nn.bias_add(fcl1, b4, name='net_pre-activation')
            fcl1 = tf.nn.relu(fcl1, name='activation')

        with tf.variable_scope('fcl_2'):
            w5 = tf.get_variable(name='_weights', shape=[8193, 4096])
            b5 = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[4096]))
            fcl2 = tf.matmul(fcl1, w5)
            fcl2 = tf.nn.bias_add(fcl2, b5, name='net_pre-activation')

        with tf.variable_scope('fcl_3'):
            w6 = tf.get_variable(name='_weights', shape=[4096, 1])
            b6 = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[1]))
            fcl3 = tf.matmul(fcl2, w6)
            self.Q_value = tf.nn.bias_add(fcl3, b6, name='net_pre-activation')

        self.error = tf.reduce_mean(tf.square(self.tf_y-self.Q_value))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.error)


    def train(self, state, action, Q_value):
        return self.session.run(self.optimizer, feed_dict={self.tf_x: state, 
                                                           self.tf_x2: action,
                                                           self.tf_y: Q_value})

    def predict(self, state, action):
        return self.session.run(self.Q_value, feed_dict={self.tf_x: state,
                                                         self.tf_x2: action})