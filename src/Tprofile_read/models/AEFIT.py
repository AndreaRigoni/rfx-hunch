

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
.##.....##..#######..########..########.##......
.###...###.##.....##.##.....##.##.......##......
.####.####.##.....##.##.....##.##.......##......
.##.###.##.##.....##.##.....##.######...##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##..#######..########..########.########
"""

class AEFIT(tf.keras.Model):

    def __init__(self, feature_dim=40, latent_dim=3, latent_intervals=None):
        super(AEFIT, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.set_model()
        
    def set_model(self):
        feature_dim = self.feature_dim
        latent_dim = self.latent_dim
        ## INFERENCE ##
        self.inference_net = tf.keras.Sequential(
            [
            tf.keras.layers.Input(shape=(feature_dim,)),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(feature_dim, activation=tf.nn.relu),
            tf.keras.layers.Dense(latent_dim * 200, activation=tf.nn.relu),
            tf.keras.layers.Dense(latent_dim * 100, activation=tf.nn.relu),
            tf.keras.layers.Dense(2*latent_dim),
            ]
        )
        ## GENERATION ##
        self.generative_net = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(latent_dim,)),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=latent_dim, activation=tf.nn.relu),
            tf.keras.layers.Dense(latent_dim * 100, activation=tf.nn.relu),
            tf.keras.layers.Dense(latent_dim * 200, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=feature_dim),
        ]
        )
        self.inference_net.build()
        self.generative_net.build()


    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, X):
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

    def plot_latent(self, z):
        s = self.decode(tf.convert_to_tensor([z]),apply_sigmoid=True) 
        x,y = tf.split(s,2,axis=1)
        plt.plot(x[0],y[0])



def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = model.compute_loss(x)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

def vae_log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)




def test_dummy(model, counts=60000, epoch=40, batch=100):
    import seaborn as sns
    from Dummy_g1data import Dummy_g1data
    dc = Dummy_g1data(counts=counts, size=20)
    # take from test set
    ts = dc.ds_array.batch(batch).make_one_shot_iterator().get_next()
    #ts = model.reparameterize(*model.encode(ts))    
    #
    # sample from random
    # ts = tf.random.normal(shape=(batch, model.latent_dim))
    #
    # sample from base
    #ts = tf.eye(model.latent_dim)

    ds = dc.ds_array.batch(batch)
    #
    count = 0
    optimizer = tf.keras.optimizers.Adam(1e-4)
    
    for e in range(epoch):
        ds = dc.ds_array.batch(batch)
        for X in ds:
                gradients, loss = compute_gradients(model, X)
                apply_gradients(optimizer, gradients, model.trainable_variables)
                count += 1

                if count % 20 == 0:
                    print('%d-%d loss: %f'%(e,count,tf.reduce_mean(loss)))
                    xy = model.reparameterize(*model.encode(ts))
                    XY  = model.decode(xy,apply_sigmoid=True)
                    X,Y = tf.split(XY,2, axis=1)

                    plt.figure(1)
                    plt.clf()
                    # for i in range(model.latent_dim):
                    for i in range(batch):
                        # sns.scatterplot(X[i],Y[i])
                        plt.plot(X[i],Y[i],'.')
                #     # plt.figure(2)
                #     # plt.clf()                
                #     # plt.plot(model._x,s[0])
                #     # sns.scatterplot(x[0],y[0])
                    plt.pause(0.001)


