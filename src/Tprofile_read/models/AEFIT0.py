

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

import tensorflow as tf
import abc

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors 

import ipysh
import Hunch_utils as Hunch


"""
..######...#######..##....##.##.....##.......##...########.
.##....##.##.....##.###...##.##.....##.....####...##.....##
.##.......##.....##.####..##.##.....##.......##...##.....##
.##.......##.....##.##.##.##.##.....##.......##...##.....##
.##.......##.....##.##..####..##...##........##...##.....##
.##....##.##.....##.##...###...##.##.........##...##.....##
..######...#######..##....##....###........######.########.
"""

class Conv1DTranspose(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, strides=1, padding='same', activation=None):
    super(Conv1DTranspose, self).__init__()
    self._layer = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(kernel_size,1), 
                                                  strides=(strides,1), padding=padding, 
                                                  activation=activation)

  def call(self, x):
    x = self._layer(x)
    return x



class MaskLayer(tf.keras.layers.Layer):
  def __init__(self, size):
    super(MaskLayer, self).__init__()
    self._size = size
    self._layers = [tf.keras.layers.Dense(1) for l in range(size)]
    for l in self._layers:
        l.build(2) # each input mask node has 2 weights

  def call(self, x):
    a = len(x.shape)-2
    l = self._layers
    x = tf.split(x,self._size,axis=a)
    y = [l[i].call(x[i]) for i in range(len(x)) ]
    return tf.concat(y, axis=a)



class NaNRandomDense(tf.keras.layers.Layer):
  def __init__(self, size):
    super(NaNRandomDense, self).__init__()
    self._size = size
    self._layers = [tf.keras.layers.Dense(1) for l in range(size)]
    for l in self._layers:
        l.build(1)

  def call(self, X):
    from random import shuffle
    l = self._layers
    def fn(x):
        shuffle(l)        
        x = tf.boolean_mask(x, tf.math.is_finite(x))
        r = x.shape[len(x.shape)-1]
        x = tf.split(tf.reshape(x,[1,-1]), r, axis=1)
        y = [ l[i].call(x[i]) for i in range(r) ]
        y = tf.concat([tf.squeeze(y), tf.zeros(self._size-len(y))], axis=0)
        return y
    if X.shape[0] is None:
        return X
    else:
        return tf.map_fn(lambda x: fn(x), X)







def tf_nan_to_num(x, num=0.):
    return tf.where(tf.math.is_nan(x), tf.ones_like(x) * num, x)





"""
.##.....##..#######..########..########.##......
.###...###.##.....##.##.....##.##.......##......
.####.####.##.....##.##.....##.##.......##......
.##.###.##.##.....##.##.....##.######...##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##..#######..########..########.########
"""

class AEFIT(tf.keras.Model):

    def __init__(self, feature_dim=20, latent_dim=20, variational_dim=10):
        super(AEFIT, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.variational_dim = variational_dim
        self._voronoi = None
        self.set_model()        
        h = tf.signal.hamming_window(11)
        h = tf.reshape(h, [int(h.shape[0]), 1, 1])
        self._x = tf.linspace(-0.4,0.4,feature_dim)
        self._h = h
        

    def set_model(self):
        latent_dim = self.latent_dim
        feature_dim = self.feature_dim
        variational_dim = self.variational_dim
        ## INFERENCE ##
        self.inference_net = tf.keras.Sequential(
            [
            tf.keras.layers.Input(shape=(2*latent_dim,)),
            NaNRandomDense(2*latent_dim),
            tf.keras.layers.Dense(2*latent_dim, activation=tf.nn.relu),
            # tf.keras.layers.Dense(30*latent_dim, activation=tf.nn.relu),
            # tf.keras.layers.Dense(20*latent_dim, activation=tf.nn.relu),
            # tf.keras.layers.Dense(10*latent_dim, activation=tf.nn.relu),
            # tf.keras.layers.Dense(feature_dim, activation=tf.nn.relu),
            # tf.keras.layers.Reshape(target_shape=(feature_dim,1,1,)),
            # Conv1DTranspose( filters=4, kernel_size=int(feature_dim/5), activation='relu' ),
            # Conv1DTranspose( filters=1, kernel_size=1),
            # tf.keras.layers.Reshape(target_shape=(feature_dim,)),
            tf.keras.layers.Dense(2*variational_dim), # no activation
            ]
        )

        # GENERATION ##
        self.generative_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(variational_dim,)),
            # tf.keras.layers.InputLayer(input_shape=(feature_dim,)),
            # tf.keras.layers.Reshape(target_shape=(feature_dim,1,)),
            # tf.keras.layers.Conv1D( filters=4, kernel_size=int(feature_dim/5), activation='relu' ),
            # tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(units=feature_dim, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=10, activation=tf.nn.relu),
            # tf.keras.layers.Dense(units=10*latent_dim, activation=tf.nn.relu),
            # tf.keras.layers.Dense(units=20*latent_dim, activation=tf.nn.relu),
            # tf.keras.layers.Dense(units=30*latent_dim, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=2*latent_dim),
        ]
        )
        self.inference_net.build()        
        self.generative_net.build()        


    def set_dataset(self, ds):
        import copy
        from sklearn.preprocessing  import normalize
        self._ds = copy.deepcopy(ds)
        self._ds.set_null(np.nan)
        self._min = np.nanmin(self._ds['prel'])
        self._max = np.nanmax(self._ds['prel'])
        self._sx = tf.linspace(self._min, self._max, self.feature_dim)
        # normalize te
        self._ds['te'] /= np.nanmax(self._ds['te'])


    def encode(self, X):
        mean, logvar = tf.split(self.inference_net(X), num_or_size_splits=2, axis=1)
        # smooth
        # s = tf.expand_dims(s, -1)        
        # s = tf.nn.conv1d(s, self._h, 1, 'SAME')
        # s = tf.squeeze(s)
        return mean, logvar        

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, s, apply_sigmoid=False):
        x = self.generative_net(s)
        if apply_sigmoid:
            x = tf.sigmoid(s)
        return x

    def get_tf_dataset(self):
        types = np.float, np.bool
        shape = ((40,),(20,))
        def gen():
            import itertools
            for i in itertools.count(0):
                if i < len(self._ds):
                    qsh = self._ds[i]
                    x,y = qsh.prel, qsh.te
                    a = tf.math.logical_and( np.isfinite(x), np.isfinite(y))
                    yield tf.concat([x, y], axis=0), a
                else:
                    return
        return tf.data.Dataset.from_generator(gen, types, shape)

    def compute_loss(self, batch_data):
        def vae_logN_pdf(sample, mean, logvar, raxis=1):
            log2pi = tf.math.log(2. * np.pi)
            return tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
        model = self
        dim = tf.cast(self.feature_dim, dtype=tf.float32)
        xy,att = batch_data
        mean,logv = model.encode(xy)
        s  = model.reparameterize(mean,logv)
        XY = model.decode(s)

        x,y = tf.split(xy,2, axis=1)
        X,Y = tf.split(XY,2, axis=1)
        s_pt = tf.boolean_mask( tf.stack([x,y], axis=2), att )
        S_pt = tf.boolean_mask( tf.stack([X,Y], axis=2), att )    
        l0 = tf.reduce_mean( tf.losses.mean_squared_error(s_pt, S_pt) )
        
        def fn(X):
            xy, XY, a = X[0],X[1],X[2]
            x,y  = tf.split(xy,2)
            X,Y  = tf.split(XY,2)
            s_pt = tf.boolean_mask( tf.stack([x,y], axis=1), a )
            S_pt = tf.boolean_mask( tf.stack([X,Y], axis=1), a )
            cxen = tf.nn.sigmoid_cross_entropy_with_logits(logits=S_pt, labels=s_pt)
            return tf.reduce_sum(cxen, axis=[0,1])
        def fn2(X):
            xy, XY, a = X[0],X[1],X[2]
            s_pt = tf.boolean_mask( xy, tf.tile(a,[2]) )
            S_pt = tf.boolean_mask( XY, tf.tile(a,[2]) )
            cxen = tf.nn.sigmoid_cross_entropy_with_logits(logits=S_pt, labels=s_pt)
            return tf.reduce_sum(cxen)

        # z_pt = tf.stack([tf.where(att, x, tf.zeros_like(x)),tf.where(att, y, tf.zeros_like(y))], axis=2)
        # Z_pt = tf.stack([tf.where(att, X, tf.zeros_like(X)),tf.where(att, Y, tf.zeros_like(Y))], axis=2)
        # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=Z_pt, labels=z_pt)
        # logpx_z =  tf.reduce_sum(cross_ent, axis=[1,2])

        logpx_z = -tf.map_fn(lambda x: fn2(x), (xy,XY,att), dtype=tf.float32)
        logpz   =  vae_logN_pdf(s, 0., 0.)
        logqz_x =  vae_logN_pdf(s, mean, logv)
        l_vae   = -tf.reduce_mean(logpx_z + logpz - logqz_x)
        
        # X1 = tf.cast( tf.clip_by_value( (X + 0.4) / 0.8 * dim, 0., dim-1), dtype = tf.int32 )
        # Y1 = tf.gather(tf.reshape(s,[-1]), X1)
        # l1 = tf.reduce_mean( tf.losses.mean_squared_error(Y, Y1) )

        # x2 = tf.boolean_mask(x,att)
        # X2 = tf.cast( (x2 + 0.4) / 0.8 * 100 , dtype = tf.int32 )
        # Y2 = tf.gather(tf.reshape(s,[-1]), X2)
        # l2 = tf.reduce_mean( tf.losses.mean_squared_error(tf.boolean_mask(y,att), Y2) )

        return  l_vae + tf.exp(l0)






"""
..#######..########..########.####.##.....##.####.########.########.########.
.##.....##.##.....##....##.....##..###...###..##.......##..##.......##.....##
.##.....##.##.....##....##.....##..####.####..##......##...##.......##.....##
.##.....##.########.....##.....##..##.###.##..##.....##....######...########.
.##.....##.##...........##.....##..##.....##..##....##.....##.......##...##..
.##.....##.##...........##.....##..##.....##..##...##......##.......##....##.
..#######..##...........##....####.##.....##.####.########.########.##.....##
"""

optimizer = tf.keras.optimizers.Adam(1e-3)

def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = model.compute_loss(x)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))


def vae_log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)



"""
.########.########.....###....####.##....##.####.##....##..######..
....##....##.....##...##.##....##..###...##..##..###...##.##....##.
....##....##.....##..##...##...##..####..##..##..####..##.##.......
....##....########..##.....##..##..##.##.##..##..##.##.##.##...####
....##....##...##...#########..##..##..####..##..##..####.##....##.
....##....##....##..##.....##..##..##...###..##..##...###.##....##.
....##....##.....##.##.....##.####.##....##.####.##....##..######..
"""



def see_test(model, batch=100):
    import seaborn as sns
    it = model.get_tf_dataset().batch(batch).make_one_shot_iterator()
    xy, att = it.get_next()
    plt.figure(1)
    plt.figure(2)
    plt.ion()
    s = model.encode(xy)
    XY = model.decode(s)    
    for i in range(len(xy)):
        x,y = tf.split(xy,2, axis=1)
        X,Y = tf.split(XY,2, axis=1)    
        plt.figure(1)
        plt.clf()
        sns.scatterplot(x[i],y[i])
        sns.scatterplot(X[i],Y[i])
        plt.pause(0.05)


def train(model, epochs=10, batch=10):
    import seaborn as sns
    from scipy import interpolate

    it = model.get_tf_dataset().batch(batch).make_one_shot_iterator()
    test_data = it.get_next()
    plt.figure(1)
    plt.figure(2)
    plt.ion()

    for epoch in range(epochs):
        count = 0        
        length = len(model._ds)
        it = model.get_tf_dataset().batch(batch).make_one_shot_iterator()
        for train_x in it:
            gradients, loss = compute_gradients(model, train_x)
            apply_gradients(optimizer, gradients, model.trainable_variables)
            count += 1

            if count % 10 == 0:
                print('%d-%d loss: %f'%(epoch,count,tf.reduce_mean(loss)))
                xy,att = test_data
                s  = model.encode(xy)
                XY = model.decode(s)
                x,y = tf.split(xy,2, axis=1)
                X,Y = tf.split(XY,2, axis=1)

                plt.figure(1)
                plt.clf()
                sns.scatterplot(x[0],y[0])
                sns.scatterplot(X[0],Y[0])
                plt.figure(2)
    
                # plt.clf()                
                # plt.plot(model._x,s[0])
                # sns.scatterplot(x[0],y[0])
                plt.pause(0.05)

