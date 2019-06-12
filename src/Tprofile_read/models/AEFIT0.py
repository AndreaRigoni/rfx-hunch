


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


class NaNMask(tf.keras.layers.Layer):
    def __init__(self, size):
        super(NaNMask, self).__init__()
        self._size = size

    def call(self, X):
        def fn(x):
            xr = tf.roll(x, shift=1, axis=0)            
            x = tf.where(tf.math.is_nan(x), xr, x)
            return x
        if X.shape[0] is None:
            return X
        else:
            return tf.map_fn(lambda x: fn(x), X)
        # Xr = tf.roll(X, shift=1, axis=1)
        # Y = tf.where(tf.math.is_nan(X), Xr, X)
        # print(Y)
        # return Y
        




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
    ''' General Autoencoder Fit Model for TF 2.0
    '''

    def __init__(self, feature_dim=40, latent_dim=2, latent_intervals=None):
        super(AEFIT, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.set_model()
        
    def set_model(self):
        feature_dim = self.feature_dim
        latent_dim = self.latent_dim
        self.nan_mask = NaNMask(feature_dim)
        ## INFERENCE ##
        self.inference_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(feature_dim,)),
            NaNRandomDense(feature_dim),
            tf.keras.layers.Dense(feature_dim, activation=tf.nn.relu),
            tf.keras.layers.Dense(latent_dim * 200, activation=tf.nn.relu),
            tf.keras.layers.Dense(latent_dim * 100, activation=tf.nn.relu),
            tf.keras.layers.Dense(2*latent_dim),
            ] )
        ## GENERATION ##
        self.generative_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(units=latent_dim, activation=tf.nn.relu),
            tf.keras.layers.Dense(latent_dim * 100, activation=tf.nn.relu),
            tf.keras.layers.Dense(latent_dim * 200, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=feature_dim),
        ] )
        self.inference_net.build()
        self.generative_net.build()

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=([-1,self.latent_dim]))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, X):
        # X = self.nan_mask(X)
        X = tf.clip_by_value(X,0.,1.)
        mean, logvar = tf.split(self.inference_net(X), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, s, apply_sigmoid=False):
        x = self.generative_net(s)
        if apply_sigmoid:
            x = tf.sigmoid(x)
        return x

    def compute_loss(self, input):
        def vae_logN_pdf(sample, mean, logvar, raxis=1):
            log2pi = tf.math.log(2. * np.pi)
            return tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
        xy = input
        mean,logv = self.encode(xy)
        z = self.reparameterize(mean,logv)
        XY = self.decode(z)
        #
        crossen =  tf.nn.sigmoid_cross_entropy_with_logits(logits=XY, labels=xy)
        logpx_z = -tf.reduce_sum(crossen, axis=[1])
        logpz   =  vae_logN_pdf(z, 0., 1.)
        logqz_x =  vae_logN_pdf(z, mean, logv)
        l_vae   = -tf.reduce_mean(logpx_z + logpz - logqz_x)
        #   
        return l_vae

    def plot_generative(self, z):
        s = self.decode(tf.convert_to_tensor([z]),apply_sigmoid=True) 
        x,y = tf.split(s,2,axis=1)
        plt.plot(x[0],y[0])

    def save(self, filename):
        self.inference_net.save_weights(filename+'_encoder.kcp')
        self.generative_net.save_weights(filename+'_decoder.kcp')

    def load(self, filename):
        self.inference_net.load_weights(filename+'_encoder.kcp')
        self.generative_net.load_weights(filename+'_decoder.kcp')
        


def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = model.compute_loss(x)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

def vae_log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)






def test_dummy(model, data=None, counts=60000, epoch=40, batch=400, loss_factor=1e-3):    
    import seaborn as sns
    import Dummy_g1data as g1
    from sklearn.cluster import KMeans
    
    fig = plt.figure('aefit_test')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # fig.tight_layout()

    if data is None:
        data = g1.Dummy_g1data(counts=counts, size=int(model.feature_dim/2), noise_var=0.)
    ts,ls = data.ds_array.batch(batch).make_one_shot_iterator().get_next()
    #ts = model.reparameterize(*model.encode(ts))    
    #
    # sample from random
    # ts = tf.random.normal(shape=(batch, model.latent_dim))
    #
    # sample from base
    #ts = tf.eye(model.latent_dim)

    kmeans = KMeans(n_clusters=model.latent_dim, random_state=0)

    def fn(x):
        xr = tf.roll(x, shift=1, axis=0)            
        x = tf.where(tf.math.is_nan(x), xr, x)
        xr = tf.roll(x, shift=1, axis=0)            
        x = tf.where(tf.math.is_nan(x), xr, x)
        xr = tf.roll(x, shift=1, axis=0)            
        x = tf.where(tf.math.is_nan(x), xr, x)
        return x

    count = 0
    optimizer = tf.keras.optimizers.Adam(loss_factor)
    g1.test_gendata(data)
    for e in range(epoch):
        ds = data.ds_array.batch(batch)
        for X in ds:
                X_data,_ = X
                X_data = tf.map_fn(lambda x: fn(x), X_data)        
                gradients, loss = compute_gradients(model, X_data)
                apply_gradients(optimizer, gradients, model.trainable_variables)
                count += 1

                if count % 20 == 0:
                    print('%d-%d loss: %f'%(e,count,tf.reduce_mean(loss)))                    
                    ts = tf.map_fn(lambda x: fn(x), ts)
                    m,v = model.encode(ts)

                    z = model.reparameterize(m,v)
                    XY  = model.decode(z,apply_sigmoid=True)
                    X,Y = tf.split(XY,2, axis=1)
                    
                    ax1.clear()
                    ax1.plot(m[:,0],m[:,1],'.')
                    # plt.plot(v[:,0],v[:,1],'.')
                    
                    ax2.clear()
                    # for i in range(model.latent_dim):
                    for i in range(batch):
                        # sns.scatterplot(X[i],Y[i])
                        ax2.plot(X[i],Y[i],'.')
                #     # plt.figure(2)
                #     # plt.clf()                
                #     # plt.plot(model._x,s[0])
                #     # sns.scatterplot(x[0],y[0])
                    fig.canvas.draw()
                    # plt.pause(0.001)




def plot_supervised_latent_distributions(model, counts=10000):
    import Dummy_g1data as g1
    dc = g1.Dummy_g1data(counts=counts, size=int(model.feature_dim/2), noise_var=0.)
    data = dc.ds_array.batch(1)
    clist=['red','blue','green','purple','grey']
    plt.figure('plot_supervised_latent_distributions')
    plt.clf()
    count = 0
    for X in data:
        ds,dl = X
        m,v = model.encode(ds)
        z   = model.reparameterize(m,v)
        #XY  = model.decode(z,apply_sigmoid=True)
        plt.plot(z[:,0],z[:,1],'.',color=clist[dl%len(clist)])
        count += 1
        if count % 100 == 0:
            plt.pause(0.001)
    