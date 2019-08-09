
'''
Prova per introdurre un set di layer convoluizonali con x e y come pixels adiacenti.
'''

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

from models.base import VAE
from models.AEFIT3 import AEFIT3




"""
.##.....##..#######..########..########.##......
.###...###.##.....##.##.....##.##.......##......
.####.####.##.....##.##.....##.##.......##......
.##.###.##.##.....##.##.....##.######...##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##..#######..########..########.########
"""

class CAEFIT(AEFIT3):
    ''' General Autoencoder Fit Model for TF 2.0
    '''
    
    def __init__(self, feature_dim=40, latent_dim=2, dprate = 0., scale=1, activation=tf.nn.relu, beta=1.):
        super(CAEFIT, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.dprate = dprate
        self.scale = scale
        self.activation = activation
        self.set_model()
        self.beta = beta
        print('CAEFIT a ready:')

    def set_model(self, training=True):
        feature_dim = self.feature_dim
        latent_dim = self.latent_dim
        if training: dprate = self.dprate
        else: dprate = 0.
        scale = self.scale
        activation = self.activation
        
    #   kernel_size: An integer or tuple/list of 3 integers, specifying the
    #   depth, height and width of the 3D convolution window.
    #   Can be a single integer to specify the same value for
    #   all spatial dimensions.

        ## INFERENCE ##
        self.inference_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(feature_dim)),
            tf.keras.layers.Reshape( target_shape=(2,int(feature_dim/2),1) ),
            tf.keras.layers.Conv2D(filters=16*scale, kernel_size=(2,3), strides=(1, 1), activation=activation, padding='SAME'),
            tf.keras.layers.Conv2D(filters=32*scale, kernel_size=(2,3), strides=(1, 1), activation=activation, padding='SAME'),
            tf.keras.layers.Conv2D(filters=32*scale, kernel_size=(2,3), strides=(1, 1), activation=activation, padding='SAME'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(feature_dim * 20 * scale, activation=activation),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(feature_dim * 20 * scale, activation=activation),            
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(feature_dim * 10 * scale, activation=activation),            
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(2*latent_dim),
            ] )

        ## GENERATION ##
        self.generative_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(units=latent_dim),
            tf.keras.layers.Dense(feature_dim * 10 * scale, activation=activation),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(feature_dim * 20 * scale, activation=activation),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(feature_dim * 20 * scale, activation=activation),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Reshape(target_shape=(2, int(feature_dim/2), int(20*scale) )),
            tf.keras.layers.Conv2DTranspose(filters=32*scale, kernel_size=(2,3), strides=(1, 1), activation=activation, padding="SAME"),
            tf.keras.layers.Conv2DTranspose(filters=32*scale, kernel_size=(2,3), strides=(1, 1), activation=activation, padding="SAME"),
            tf.keras.layers.Conv2DTranspose(filters=16*scale, kernel_size=(2,3), strides=(1, 1), activation=activation, padding="SAME"),
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1,1), strides=(1, 1), padding="SAME"),
            tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(units=feature_dim),
        ] )
        self.inference_net.build()
        self.generative_net.build()

        self.compile(  
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss = self.vae3_loss,
            metrics = ['accuracy', self.sce, self.akl, self.kld]
        )


#     def sample(self, eps=None):
#         if eps is None:
#             eps = tf.random.normal(shape=([-1,self.latent_dim]))
#         return self.decode(eps, apply_sigmoid=True)

#     def encode(self, X):
#         X = tf.clip_by_value(X,0.,1.)
#         mean, logvar = tf.split(self.inference_net(X), num_or_size_splits=2, axis=1)
#         return mean, logvar

#     def reparameterize(self, mean, logvar):
#         eps = tf.random.normal(shape=mean.shape)
#         return eps * tf.exp(logvar * .5) + mean

#     def decode(self, s, apply_sigmoid=False):
#         x = self.generative_net(s)
#         if apply_sigmoid:
#             x = tf.sigmoid(x)
#         return x

#     def compute_loss(self, input):
#         def vae_logN_pdf(sample, mean, logvar, raxis=1):
#             log2pi = tf.math.log(2. * np.pi)
#             return tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
#         #
#         att = tf.math.is_nan(input)
#         xy  = tf_nan_to_num(input, 0.)
#         mean,logv = self.encode(xy)
#         z = self.reparameterize(mean,logv)
#         XY = self.decode(z)
#         XY = tf.where(att, tf.zeros_like(XY), XY)
#         #
#         crossen =  tf.nn.sigmoid_cross_entropy_with_logits(logits=XY, labels=xy)
#         logpx_z = -tf.reduce_sum(crossen, axis=[1])
#         logpz   =  vae_logN_pdf(z, 0., 1.)
#         logqz_x =  vae_logN_pdf(z, mean, logv)
#         l_vae   = -tf.reduce_mean(self.beta * logpx_z + logpz - logqz_x)/self.beta
#         #
#         return l_vae

#     def plot_generative(self, z):
#         s = self.decode(tf.convert_to_tensor([z]),apply_sigmoid=True) 
#         x,y = tf.split(s,2,axis=1)
#         plt.plot(x[0],y[0])

#     def save(self, filename):
#         self.inference_net.save_weights(filename+'_encoder.kcp')
#         self.generative_net.save_weights(filename+'_decoder.kcp')

#     def load(self, filename):
#         self.inference_net.load_weights(filename+'_encoder.kcp')
#         self.generative_net.load_weights(filename+'_decoder.kcp')
        


# def compute_gradients(model, x):
#     with tf.GradientTape() as tape:
#         loss = model.compute_loss(x)
#     return tape.gradient(loss, model.trainable_variables), loss

# def apply_gradients(optimizer, gradients, variables):
#     optimizer.apply_gradients(zip(gradients, variables))







# def test_dummy(model, data, epoch=40, batch=400, loss_factor=1e-3):
#     import seaborn as sns
#     import Dummy_g1data as g1
#     import itertools
#     from sklearn.cluster import KMeans
    
#     fig = plt.figure('aefit_test')
#     ax1 = fig.add_subplot(121)
#     ax2 = fig.add_subplot(122)
#     ds_size = len(data)

#     count = 0
#     optimizer = tf.keras.optimizers.Adam(loss_factor)
#     g1.test_gendata(data)
#     for e in range(epoch):
#         ds = data.ds_array.batch(batch)
#         for X in ds:
#                 X_data,_ = X                
#                 gradients, loss = compute_gradients(model, X_data)
#                 apply_gradients(optimizer, gradients, model.trainable_variables)
#                 count += 1

#                 if count % 20 == 0:
#                   print('%d-%d loss: %f'%(e,count,tf.reduce_mean(loss)))                    
                  
#                   m,v = model.encode(X_data)
#                   z   = model.reparameterize(m,v)
#                   XY  = model.decode(z,apply_sigmoid=True)
#                   X,Y = tf.split(XY,2, axis=1)
                  
#                   ax1.clear()                  
#                   ax1.plot(m[:,0],m[:,1],'.')
#                   # plt.plot(v[:,0],v[:,1],'.')
                  
#                   ax2.clear()
#                   for i in range(batch):
#                       ax2.plot(X[i],Y[i],'.')

#                   fig.canvas.draw()


