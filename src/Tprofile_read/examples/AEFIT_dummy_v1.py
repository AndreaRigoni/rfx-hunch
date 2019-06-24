
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
import models.AEFIT_v1 as aefit

ipysh.Bootstrap_support.debug()

# sess = tf.Session()
config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=8, \
                        allow_soft_placement=True, device_count = {'CPU': 8})
sess = tf.InteractiveSession(config=config)


# feature_dim = 40
# def net(x):
#     x = tf.reshape(x, shape=[-1, feature_dim])
#     # Convolution Layer 
#     x = tf.layers.dense(x, units=200, activation=tf.nn.relu)
#     x = tf.layers.dense(x, units=200, activation=tf.nn.relu)
#     x = tf.layers.dense(x, units=200, activation=tf.nn.relu)
#     x = tf.layers.dense(x, units=200, activation=tf.nn.relu)
#     x = tf.layers.dense(x, units=200, activation=tf.nn.relu)
#     x = tf.layers.dense(x, units=200, activation=tf.nn.relu)
#     x = tf.layers.dense(x, units=200, activation=tf.nn.relu)
#     return x


# # Currently we are only allowed to create 1 profiler per process.
# profiler = tf.profiler.Profiler(sess.graph)
# option_builder = tf.profiler.ProfileOptionBuilder

# ALL_ADVICE = {
#     'ExpensiveOperationChecker': {},
#     'AcceleratorUtilizationChecker': {},
#     'JobChecker': {},  # Only available internally.
#     'OperationChecker': {},
# }

# # Initialize the variables (i.e. assign their default value)

# ds = dummy.Dummy_g1data(counts=60000)
# ds = ds.get_tf_dataset_array().batch(1000)
# ds = ds.prefetch(1000)
# it = ds.make_one_shot_iterator()
# # x,l = sess.run(it.get_next())
# x,l = it.get_next()


# def profile(sess=sess, steps=10):
#         y = net(x)

#         init = tf.global_variables_initializer()
#         sess.run(init)
        
#         for i in range(steps):
#                 run_meta = tf.compat.v1.RunMetadata()
#                 _ = sess.run(y,
#                         options=tf.compat.v1.RunOptions(
#                         trace_level=tf.RunOptions.FULL_TRACE),
#                         run_metadata=run_meta)
#                 profiler.add_step(i, run_meta)

#                 # Profile the parameters of your model.
#                 profiler.profile_name_scope(options=(option_builder.trainable_variables_parameter()))

#                 # Or profile the timing of your model operations.
#                 opts = option_builder.time_and_memory()
#                 profiler.profile_operations(options=opts)

#                 # Or you can generate a timeline:
#                 opts = (option_builder(option_builder.time_and_memory())
#                                 .with_step(i)
#                                 .with_timeline_output('profile').build())
#                 profiler.profile_graph(options=opts)
#                 # Auto detect problems and generate advice.
#                 # profiler.advise(options=ALL_ADVICE)



ds = dummy.Dummy_g1data(counts=60000)
m = aefit.AEFIT_v1()