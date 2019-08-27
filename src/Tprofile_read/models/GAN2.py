
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

import models.AEFIT5


"""
..######......###....##....##
.##....##....##.##...###...##
.##.........##...##..####..##
.##...####.##.....##.##.##.##
.##....##..#########.##..####
.##....##..##.....##.##...###
..######...##.....##.##....##
"""


class GAN2(models.AEFIT5.AEFIT5):
    def __init__(self, *args, **kwargs):
        super(GAN2, self).__init__(*args, **kwargs)
        ## DISCRIMINATOR ##        
        self.discriminator_net = tf.keras.Sequential([            
            tf.keras.layers.Input(shape=(self.latent_dim,)),
            tf.keras.layers.Dense(1),
        ])
        self.discriminator_net.build()
        self.set_optimizers()
        print('GAN2 ready:')

    def get_learning_rate(self):
        return [ x._get_hyper('learning_rate') for x in self.optimizer ]

    def set_learning_rate(self, lf):
        for x in self.optimizer:
            x._set_hyper('learning_rate', lf)

    def set_optimizers(self, opt=tf.keras.optimizers.Adam, dfactor=1e-5):
        self.optimizer = [
            opt(dfactor),
            opt(dfactor)
        ]



    @tf.function
    def encode(self, x, training=False):
        me,lv = tf.split(self.inference_net(x, training=training), num_or_size_splits=2, axis=1)
        return me, lv

    @tf.function
    def decode(self, z, apply_sigmoid=False, training=False):
        y = self.generative_net(z, training=training)
        return y
    
    @tf.function
    def reparametrize(self, mean, logvar = 0.):
        eps = tf.random.normal(shape=mean.shape) * self.beta
        return eps * tf.exp(logvar * .5) + mean

    @tf.function
    def discriminate(self, x, training=True):
        s,_ = self.encode(x, training=training)
        return self.discriminator_net(s, training=training)


    def call(self, data, training=True):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        @tf.function
        def discriminator_loss(real_output, fake_output):
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            total_loss = real_loss + fake_loss
            return total_loss

        @tf.function
        def generator_loss(fake_output):        
            return cross_entropy(tf.ones_like(fake_output), fake_output)
        
        s,l = self.encode(data)
        z   = self.reparametrize(s,l)
        # kl_loss = -0.5 * tf.reduce_sum(lv + 1 - tf.square(me) + tf.exp(lv))
        generated_data = self.decode(z, training=training)
        real_output = self.discriminate(data, training=training)
        fake_output = self.discriminate(generated_data, training=training)
        self.add_loss( lambda: generator_loss(fake_output), inputs=True )
        self.add_loss( lambda: discriminator_loss(real_output, fake_output), inputs=True )
        
        return generated_data


    @tf.function
    def train_step(self, data, training=True):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        @tf.function
        def discriminator_loss(real_output, fake_output):
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            total_loss = real_loss + fake_loss
            return total_loss

        @tf.function
        def generator_loss(fake_output):        
            return cross_entropy(tf.ones_like(fake_output), fake_output)
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            s,l = self.encode(data)
            z   = self.reparametrize(s,l)
            # kl_loss = -0.5 * tf.reduce_sum(lv + 1 - tf.square(me) + tf.exp(lv))
            generated_data = self.decode(z, training=training)
            real_output = self.discriminate(data, training=training)
            fake_output = self.discriminate(generated_data, training=training)
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)
        
            # self.call(data, training=training)
            # gen_loss  = self.losses[0]
            # disc_loss = self.losses[1]

        if training is True:
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generative_net.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.inference_net.trainable_variables 
                                                                        + self.discriminator_net.trainable_variables)
            self.optimizer[0].apply_gradients(zip(gradients_of_generator, self.generative_net.trainable_variables))
            self.optimizer[1].apply_gradients(zip(gradients_of_discriminator, self.inference_net.trainable_variables 
                                                                              + self.discriminator_net.trainable_variables))

        return tf.reduce_sum([gen_loss, disc_loss])

    






