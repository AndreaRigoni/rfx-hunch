
'''
GAN First attempt
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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

from models.base import THunchModel
from models.base import GAN
from models.base import Dataset


"""
..######......###....##....##
.##....##....##.##...###...##
.##.........##...##..####..##
.##...####.##.....##.##.##.##
.##....##..#########.##..####
.##....##..##.....##.##...###
..######...##.....##.##....##
"""

class GAN3(GAN):

    def __init__(self, feature_dim=40, latent_dim=2, dprate = 0., scale=1, activation=tf.nn.relu, beta=1.):
        # self.set_optimizers()
        super(GAN3, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.dprate = dprate
        self.scale = scale
        self.beta = beta
        self.activation = activation
        self.stop_training = False
        self.set_model()

        self.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss = tf.keras.losses.mse
        )

        print('GAN3 ready:')


    # def get_learning_rate(self):
    #     return [ x._get_hyper('learning_rate') for x in self.optimizer ]

    # def set_learning_rate(self, lf):
    #     for x in self.optimizer:
    #         x._set_hyper('learning_rate', lf)

    # def set_optimizers(self, opt=tf.keras.optimizers.Adam, dfactor=1e-3):
    #     self.optimizer = [
    #         opt(dfactor),
    #         opt(dfactor)
    #     ]

    def set_model(self, training=True):
        feature_dim = self.feature_dim
        latent_dim = self.latent_dim
        if training: dprate = self.dprate
        else: dprate = 0.
        scale = self.scale
        activation = self.activation
        
        ## INFERENCE ##
        self.inference_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(feature_dim)),
            # tf.keras.layers.Reshape( target_shape=(2,int(feature_dim/2),1) ),
            # tf.keras.layers.Conv2D(filters=32*scale, kernel_size=(2,3), strides=(1, 1), padding='SAME', activation=activation),
            # tf.keras.layers.Dropout(dprate),            
            # tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(feature_dim * 20 * scale, activation=activation),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(feature_dim * 20 * scale, activation=activation),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(feature_dim * 10 * scale, activation=activation),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(feature_dim * 10 * scale, activation=activation),
            tf.keras.layers.Dense(1),
        ] )

        # # ## GENERATION ##
        self.generative_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(latent_dim,)),            
            tf.keras.layers.Dense(latent_dim),
            tf.keras.layers.Dense(feature_dim * 10 * scale, activation=activation),
            tf.keras.layers.Dense(feature_dim * 10 * scale, activation=activation),
            tf.keras.layers.Dense(feature_dim * 20 * scale, activation=activation),
            tf.keras.layers.Dense(feature_dim * 20 * scale, activation=activation),
            # tf.keras.layers.Reshape(target_shape=(2, int(feature_dim/2), int(20*scale) )),
            # tf.keras.layers.Conv2DTranspose(filters=32*scale, kernel_size=(2,3), strides=(1, 1), padding="SAME", activation=activation),
            # tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1,1), strides=(1, 1), padding="SAME"),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(feature_dim),
        ] )

        self.inference_net.build()
        self.generative_net.build()


    def encode(self, x, training=True):
        return self.inference_net(x, training=training)

    def decode(self, s, apply_sigmoid=True, training=True):
        y = self.generative_net(s, training=training)
        return y

    def discriminate(self, x, training=True):
        return self.encode(x, training=training)        


    def call(self, xy, training=True):
        batch_size = tf.shape(xy)[0]
        noise = tf.random.normal([batch_size, self.latent_dim], stddev=self.beta)
        generated_data = self.generative_net(xy + noise, training=training)            
        return generated_data
        

    # def gan_loss_1(self, xy, XY):
    #     pass


    # def gan_loss(self, xy, XY):
    #     cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    #     def discriminator_loss(real_output, fake_output):
    #         real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    #         fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    #         total_loss = real_loss + fake_loss
    #         return total_loss

    #     def generator_loss(fake_output):        
    #         return cross_entropy(tf.ones_like(fake_output), fake_output)

    #     real_output = self.discriminate(xy, training=True)
    #     fake_output = self.discriminate(generated_data, training=training)
    #     gen_loss  = generator_loss(fake_output)
    #     disc_loss = discriminator_loss(real_output, fake_output)
    #     return gen_loss + disc_loss
        

    @tf.function
    def train_step(self, data, training=True):
        generator_optimizer = self.optimizer
        discriminator_optimizer = self.optimizer
        input_data,real_data = data[0]
        
        batch_size = tf.shape(input_data)[0]
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        def discriminator_loss(real_output, fake_output):
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            total_loss = real_loss + fake_loss
            return total_loss

        def generator_loss(fake_output):        
            return cross_entropy(tf.ones_like(fake_output), fake_output)
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            noise = tf.random.normal([batch_size, self.latent_dim], stddev=self.beta)
            generated_data = self.generative_net(input_data + noise, training=True)
            real_output = self.discriminate(real_data, training=True)
            fake_output = self.discriminate(generated_data, training=True)
            gen_loss  = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        if training:
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generative_net.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.inference_net.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generative_net.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.inference_net.trainable_variables))
        
        return tf.reduce_sum([gen_loss, disc_loss])

    

                
            







