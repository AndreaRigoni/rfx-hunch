from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import abc



class VAE(tf.keras.Model):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        super(VAE, self).__init__()

    @abc.abstractmethod
    def encode(self, x, training=False):
        return NotImplemented

    @abc.abstractmethod
    def decode(self, s, apply_sigmoid=False, training=False):
        return NotImplemented

    @abc.abstractmethod
    def reparametrize(self, x, training=True):
        return NotImplemented



class GAN(tf.keras.Model):    
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        super(GAN, self).__init__()

    @abc.abstractmethod
    def encode(self, x, training=False):
        return NotImplemented

    @abc.abstractmethod
    def decode(self, s, apply_sigmoid=False, training=False):
        return NotImplemented

    @abc.abstractmethod
    def discriminate(self, x, training=True):
        return NotImplemented



class Dataset():
    __metaclass__ = abc.ABCMeta
    # BASE PROPERTIES    
    @property
    @abc.abstractmethod
    def ds_tuple(self):
        return NotImplemented

    @property
    @abc.abstractmethod
    def ds_array(self):
        return NotImplemented






def test_dummy(model, data, epoch=40, batch=400, loss_factor=1e-3):
    import itertools
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as colors 

    def compute_gradients(model, x):
        with tf.GradientTape() as tape:
            loss = model.compute_loss(x)
        return tape.gradient(loss, model.trainable_variables), loss

    def apply_gradients(optimizer, gradients, variables):
        optimizer.apply_gradients(zip(gradients, variables))

    fig = plt.figure('models_test_dummy')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    count = 0
    optimizer = tf.keras.optimizers.Adam(loss_factor)
    for e in range(epoch):
        ds = data.ds_array.batch(batch)
        for X in ds:
                X_data,_ = X
                gradients, loss = compute_gradients(model, X_data)
                apply_gradients(optimizer, gradients, model.trainable_variables)
                count += 1

                if count % 20 == 0:
                  print('%d-%d loss: %f'%(e,count,tf.reduce_mean(loss)))                    
                  
                  m,v = model.encode(X_data)
                  z   = model.reparameterize(m,v)
                  XY  = model.decode(z,apply_sigmoid=True)
                  X,Y = tf.split(XY,2, axis=1)
                  
                  ax1.clear()                  
                  ax1.plot(m[:,0],m[:,1],'.')
                  # plt.plot(v[:,0],v[:,1],'.')
                  
                  ax2.clear()
                  for i in range(batch):
                      ax2.plot(X[i],Y[i],'.')

                  fig.canvas.draw()


