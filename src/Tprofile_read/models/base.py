from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors 

import numpy as np
import tensorflow as tf
import abc

import models

# !pip install --user nbmultitask
# from nbmultitask import ThreadWithLogAndControls
# from nbmultitask import ProcessWithLogAndControls
# from IPython.display import clear_output


class THunchModel(tf.keras.Model):
    __metaclass__ = abc.ABCMeta
    def __init__(self, *args, **kwargs):
        super(THunchModel, self).__init__(*args, **kwargs)
        self.stop_training = False


    @abc.abstractmethod
    # @learning_rate.getter
    def get_learning_rate(self):
        return self.optimizer._get_hyper('learning_rate')

    @abc.abstractmethod
    # @learning_rate.setter
    def set_learning_rate(self, lf):
        self.optimizer._set_hyper('learning_rate', lf)

    learning_rate = property(fget=lambda self: self.get_learning_rate(), 
                             fset=lambda self, value: self.set_learning_rate(value))

    @abc.abstractmethod
    def train_step(self, x, training=True):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, dataset, epoch=10, batch=200, learning_rate=None):
        if issubclass(type(dataset), models.base.Dataset):
            dataset = dataset.ds_array
        ds = dataset.take(batch)
        dt = dataset.skip(batch)

        if learning_rate is not None:
            self.learning_rate = learning_rate

        def compute_test_loss():
            ts_loss = 0
            for data in ds.batch(batch, drop_remainder=True):
                ts_loss = self.train_step(data,training=False)
            return ts_loss.numpy()

        for e in range(epoch):
            import sys
            print('EPOCH: ',e)
            for i,data in enumerate(dt.batch(batch, drop_remainder=True)):
                loss = self.train_step(data)
                sys.stdout.write('it: %d - loss: %f \r'%(i,loss))
                if self.stop_training:
                    break
            if self.stop_training:
                print(' Train early stopping ... ')
                break
            print('test loss: ',compute_test_loss())

    @abc.abstractmethod
    def train_thread(self, data, epoch=10, batch=200, learning_rate=None):
        from nbmultitask import ThreadWithLogAndControls
        from nbmultitask import ProcessWithLogAndControls
        class AsyncTrain(ThreadWithLogAndControls):
            def __init__(self, model, data, **kwargs):
                self._model = model
                model.stop_training = False
                fn = lambda thread_print: model.train(data, **kwargs)
                super(AsyncTrain, self).__init__(target=fn, name='async train')
    
            def terminate(self):
                self._model.stop_training = True
        task = AsyncTrain(self, data, epoch=epoch, batch=batch, learning_rate=learning_rate)
        return task


   


class VAE(THunchModel):
    __metaclass__ = abc.ABCMeta
    def __init__(self, *args, **kwargs):
        super(VAE, self).__init__(*args, **kwargs)

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





class GAN(THunchModel):    
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        super(GAN, self).__init__()

    @abc.abstractmethod
    def encode(self, x, training=False):
        return NotImplementedError

    @abc.abstractmethod
    def decode(self, s, apply_sigmoid=False, training=False):
        return NotImplementedError

    @abc.abstractmethod
    def discriminate(self, x, training=True):
        return NotImplementedError




class Dataset():
    __metaclass__ = abc.ABCMeta

    # BASE PROPERTIES    
    @property
    @abc.abstractmethod
    def ds_tuple(self):
        return NotImplementedError

    @property
    @abc.abstractmethod
    def ds_array(self):
        return NotImplementedError








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


class PlotLearning(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []
        plt.ion()
        plt.show()

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        _, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        # clear_output(wait=True)
        plt.clf()
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()

        plt.show()


tf.keras.Model

def train(model, data, validation_data=None, epoch=40, batch=200, learning_rate=1e-3, log_name=None, callbacks=None):     
    model.learning_rate = learning_rate
    if issubclass(type(data), models.base.Dataset):
        data = data.ds_array
    if batch: data = data.batch(batch, drop_remainder=True)
    if issubclass(type(model), models.base.VAE):
        data = data.map(lambda x,y: (x,x))
    if callbacks is None:
        callbacks = [LearningRatePowDecay(), ResetCallback()]
    history = model.fit(data, epochs=epoch, callbacks=callbacks + tensorboard_log(log_name), verbose=1) 
    return history
    
def train_thread(model, data, epoch=40, batch=200, learning_rate=1e-3, log_name=None, callbacks=None):     
    from nbmultitask import ThreadWithLogAndControls
    from nbmultitask import ProcessWithLogAndControls
    class AsyncTrain(ThreadWithLogAndControls):
        def __init__(self, model, data, **kwargs):
            self._model = model
            model.stop_training = False
            fn = lambda thread_print: train(model, data, **kwargs)
            super(AsyncTrain, self).__init__(target=fn, name='async train')

        def terminate(self):
            self._model.stop_training = True
    task = AsyncTrain(model, data, epoch=epoch, batch=batch, learning_rate=learning_rate, log_name=log_name, callbacks=callbacks)
    return task




## DISCOURAGE THIS ##
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
                  m,v = model.encode(X_data, training=False)
                  z   = model.reparametrize(m,v)
                  XY  = model.decode(z, training=False, apply_sigmoid=True)
                  X,Y = tf.split(XY,2, axis=1)
                  
                  ax1.clear()
                  ax1.plot(m[:,0],m[:,1],'.')
                  # plt.plot(v[:,0],v[:,1],'.')
                  
                  ax2.clear()
                  for i in range(batch):
                      ax2.plot(X[i],Y[i],'.')

                  fig.canvas.draw()




