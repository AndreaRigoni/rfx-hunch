


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
import models
# from models.base import VAE


"""
.##....##....###....##....##....########..########.##....##..######..########
.###...##...##.##...###...##....##.....##.##.......###...##.##....##.##......
.####..##..##...##..####..##....##.....##.##.......####..##.##.......##......
.##.##.##.##.....##.##.##.##....##.....##.######...##.##.##..######..######..
.##..####.#########.##..####....##.....##.##.......##..####.......##.##......
.##...###.##.....##.##...###....##.....##.##.......##...###.##....##.##......
.##....##.##.....##.##....##....########..########.##....##..######..########
"""



class NaNDense(tf.keras.layers.Dense):
    """Just your regular densely-connected NN layer.
    """
    def __init__(self,
               units,
               activation=None,
               use_bias=True,
               **kwargs):
        super(NaNDense, self).__init__( units, activation, **kwargs)
        
    
    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        inputs = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        outputs = tf.matmul(inputs, self.kernel)
        if self.use_bias: 
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


# class Reparametrize1D(tf.keras.layers.Layer):
#     """ VAE REPARAMETRIZATION LAYER
#     """
#     def __init__(self, **kwargs):
#         super(Reparametrize1D, self).__init__(**kwargs)
    
#     @tf.function
#     def reparametrize(self, z_mean, z_log_var):
#         batch = tf.shape(z_mean)[0]
#         dim   = tf.shape(z_mean)[1]
#         epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
#         return z_mean + tf.exp(0.5 * z_log_var) * epsilon

#     @tf.function
#     def call(self, inputs, training=True):
#         inputs = tf.convert_to_tensor(inputs)
#         mean, logvar = tf.split(inputs, num_or_size_splits=2, axis=1)
#         akl_loss = -0.5 * tf.reduce_sum(1. + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
#         mean = self.reparametrize(mean,logvar)
#         self.add_loss( akl_loss, inputs=True )
#         return mean
        
    
    


"""
.##.....##..#######..########..########.##......
.###...###.##.....##.##.....##.##.......##......
.####.####.##.....##.##.....##.##.......##......
.##.###.##.##.....##.##.....##.######...##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##..#######..########..########.########
"""
class AEFIT5(models.base.VAE):
    ''' General Autoencoder Fit Model for TF 2.0
    '''    
    def __init__(self, feature_dim=40, latent_dim=2, dprate = 0., activation=tf.nn.relu, beta=1., 
                 encoder_geometry=[20,20,10,10], decoder_geometry=None, scale=1, *args, **kwargs):
        self.super = super(AEFIT5, self)
        self.super.__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim        
        self.dprate = dprate
        self.scale = scale
        self.activation = activation
        self.beta = beta 
        self.apply_sigmoid = False
        self.bypass = False

        if decoder_geometry is None: decoder_geometry = encoder_geometry[::-1] # reverse order
        inference_net, generative_net = self.set_model(feature_dim, latent_dim, dprate=dprate, scale=scale, activation=activation,
                                                       encoder_geometry=encoder_geometry, decoder_geometry=decoder_geometry)
        self.inference_net = inference_net
        self.generative_net = generative_net
        self.build(input_shape=inference_net.input_shape)
        self.compile(
            optimizer  = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss       = self.compute_cross_entropy_loss,
            logit_loss = True,
            metrics    = ['accuracy']
        )
        print('AEFIT5 ready:')


    
    def set_model(self, feature_dim, latent_dim, dprate = 0., activation=tf.nn.relu, 
                  encoder_geometry=[20,20,10], decoder_geometry=[10,20,20], scale=1):
        def add_dense_encode(self, fdim=feature_dim, ldim=latent_dim, geometry=[20,20,10,10]):
            for _,size in enumerate(geometry):
                self.add(tf.keras.layers.Dense(fdim*size*scale, activation=activation))
                self.add(tf.keras.layers.Dropout(dprate))
            self.add(tf.keras.layers.Dense(ldim))
            return self

        def add_dense_decode(self, fdim=feature_dim, ldim=latent_dim, geometry=[10,10,20,20]):            
            for _,size in enumerate(geometry):
                self.add(tf.keras.layers.Dense(fdim*size*scale, activation=activation))
                self.add(tf.keras.layers.Dropout(dprate))
            self.add(tf.keras.layers.Dense(fdim))
            return self
        # add methods to Sequential class
        tf.keras.Sequential.add_dense_encode = add_dense_encode
        tf.keras.Sequential.add_dense_decode = add_dense_decode
        
        ## INFERENCE ##
        inference_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(feature_dim,)),
            NaNDense(feature_dim),
        ]).add_dense_encode(ldim=2*latent_dim, geometry=encoder_geometry)

        ## GENERATION ##
        generative_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(latent_dim)
        ]).add_dense_decode(geometry=decoder_geometry)
        
        return inference_net, generative_net

    @tf.function
    def reparametrize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    @tf.function
    def encode(self, X, training=None):
        mean, logvar = tf.split(self.inference_net(X, training=training), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def decode(self, s, training=None, apply_sigmoid=False):
        x = self.generative_net(s, training=training)        
        if training or not self.apply_sigmoid: return x
        else                                 : return tf.sigmoid(x)

    def call(self, xy, training=None):
        att = tf.math.is_nan(xy)
        xy  = tf.where(att, tf.zeros_like(xy), xy)
        if self.bypass == False:
            mean, logvar = self.encode(xy, training=training)
            z = self.reparametrize(mean,logvar)
            XY = self.decode(z, training=training)
            XY  = tf.where(att, tf.zeros_like(XY), XY)
            kl_loss = -0.5 * tf.reduce_sum(1. + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
        else:
            XY = xy
            kl_loss = 0.
        self.add_loss( lambda: self.beta * kl_loss )
        return XY

    def compile(self, optimizer=None, loss=None, logit_loss=False, metrics=None, **kwargs):
        if optimizer is None: 
            if self.optimizer: optimizer = self.optimizer
            else             : optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        if loss is None: loss_wrapper = self.loss
        else: 
            self.apply_sigmoid = logit_loss
            loss_wrapper = lambda xy,XY: loss( tf.where(tf.math.is_nan(xy), tf.zeros_like(xy), xy), XY)        
        return self.super.compile(optimizer, loss=loss_wrapper, metrics=metrics, **kwargs)

    def compute_cross_entropy_loss(self, xy, XY):
        crossen =  tf.nn.sigmoid_cross_entropy_with_logits(logits=XY, labels=xy)
        logpx_z =  tf.reduce_sum( crossen , axis=1)
        return logpx_z

    def recover(self,x):
        xr = self.call(x, training=False)
        return tf.where(tf.math.is_nan(x),xr,x)







    