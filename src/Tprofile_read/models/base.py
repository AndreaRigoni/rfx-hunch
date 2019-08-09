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
        self.stop_training = False

    @abc.abstractmethod
    def encode(self, x, training=False):
        return NotImplemented

    @abc.abstractmethod
    def decode(self, s, apply_sigmoid=False, training=False):
        return NotImplemented

    @abc.abstractmethod
    def reparametrize(self, x, training=True):
        return NotImplemented    

    def save(self, filename):
        self.inference_net.save_weights(filename+'_encoder.kcp')
        self.generative_net.save_weights(filename+'_decoder.kcp')

    def load(self, filename):
        self.inference_net.load_weights(filename+'_encoder.kcp')
        self.generative_net.load_weights(filename+'_decoder.kcp')





class GAN(tf.keras.Model):    
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        super(GAN, self).__init__()
        self.stop_training = False

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









def tensorboard_log(name=None):
    import datetime
    if name is None: return []
    else           : name = name + '_'
    log_base_dir = ipysh.abs_srcdir+"/jpnb/logs"
    log_dir  = log_base_dir+"/fit/" + name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_loss = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')
    
    cpt_file = log_dir + '/models.ckpt'
    check_pt = tf.keras.callbacks.ModelCheckpoint(filepath=cpt_file, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
    
    return [ batch_loss, check_pt ]

class ResetCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_states
    def set_model(self, model):
        self.model = model

class LearningRatePowDecay(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, epoch_drop=0.8, verbose=True):
        self.schedule = lambda e,lr: lr * pow(0.9, e)
        self.verbose  = verbose    
    
    
def train(model, data, epoch=40, batch=200, loss_factor=1e-3, log_name=None, callbacks=None):     
    model.loss_factor = loss_factor
    ds = data.ds_array.batch(batch).map(lambda x,y: (x,x))
    ds = ds.take( int(len(data)/batch) )
    history = model.fit(ds, epochs=epoch, callbacks=[LearningRatePowDecay(), ResetCallback()] + tensorboard_log(log_name), verbose=1) 
    return history
    
def train_thread(model, data, epoch=40, batch=200, loss_factor=1e-3, log_name=None, callbacks=None):     
    from threading import Thread
    t = Thread(target=train, args=(model, data, epoch, batch, loss_factor, log_name, callbacks))
    return t


    
def manual_train(model, data, epoch=5, batch=200, loss_factor=1e-3):
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
                  m,v = model.encode(X_data, train=False)
                  z   = model.reparameterize(m,v)
                  XY  = model.decode(z, train=False, apply_sigmoid=True)
                  X,Y = tf.split(XY,2, axis=1)
                  
                  ax1.clear()                  
                  ax1.plot(m[:,0],m[:,1],'.')
                  # plt.plot(v[:,0],v[:,1],'.')
                  
                  ax2.clear()
                  for i in range(batch):
                      ax2.plot(X[i],Y[i],'.')

                  fig.canvas.draw()


