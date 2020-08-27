import tensorflow as tf
import tensorflow.layers as layers
import tensorflow.nn as nn
import numpy as np
import environment

num_of_feature = 10
filter_size=3
memory_size = 20
num_of_asset= 10

X=tf.placeholder(tf.float32,[None,10,50,4],name="s")
M=tf.placeholder(tf.float32,[None,20,10,1],name='M')

conv1 = layers.conv2d(X, 2, [1,3], padding='VALID')

envi=environment.env()
s = envi.start()
s = np.expand_dims(s,axis=0)
s = np.reshape(s,(1,10,50,4))

memory = np.zeros([memory_size,num_of_asset,1],dtype = np.float32)
memory = np.expand_dims(memory,axis=0)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    con = sess.run(conv1, {X: s, M: memory})
    print(con.shape)