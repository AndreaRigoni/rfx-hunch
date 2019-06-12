
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors 

import ipysh
import Hunch_utils  as Htls
import Hunch_lsplot as Hplt

import Dummy_g1data as dummy

ipysh.Bootstrap_support.debug()

# sess = tf.Session()
sess = tf.InteractiveSession()


feature_dim = 40
def net(x):
    x = tf.reshape(x, shape=[-1, feature_dim])
    # Convolution Layer 
    x = tf.layers.dense(x, units=200, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=200, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=200, activation=tf.nn.relu)
    return x


ds = dummy.Dummy_g1data(counts=60000)
it = ds.get_tf_dataset_array().batch(200).make_one_shot_iterator()
x,l = it.get_next()

y = net(x)


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
sess.run(init)


# for i in range(100):
#     sess.run(y)


