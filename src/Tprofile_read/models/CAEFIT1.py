
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

class CAEFIT1(AEFIT3):
    ''' General Autoencoder Fit Model for TF 2.0
    '''
    
    def __init__(self, feature_dim=40, latent_dim=2, dprate = 0., scale=1, activation=tf.nn.relu, beta=1.):
        super(CAEFIT1, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.dprate = dprate
        self.scale = scale
        self.activation = activation
        self.set_model()
        self.beta = beta
        print('CAEFIT1 ready:')

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
            tf.keras.layers.Dense(feature_dim * 10 * scale, use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False, center=True),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(feature_dim * 20 * scale, use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False, center=True),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(feature_dim * 20 * scale, use_bias=False),
            tf.keras.layers.BatchNormalization(scale=False, center=True),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Reshape(target_shape=(2, int(feature_dim/2), int(20*scale) )),
            tf.keras.layers.Conv2DTranspose(filters=32*scale, kernel_size=(2,3), strides=(1, 1), use_bias=False, padding="SAME"),
            tf.keras.layers.BatchNormalization(scale=False, center=True),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2DTranspose(filters=32*scale, kernel_size=(2,3), strides=(1, 1), use_bias=False, padding="SAME"),
            tf.keras.layers.BatchNormalization(scale=False, center=True),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2DTranspose(filters=16*scale, kernel_size=(2,3), strides=(1, 1), use_bias=False, padding="SAME"),
            tf.keras.layers.BatchNormalization(scale=False, center=True),
            tf.keras.layers.Activation('relu'),
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

