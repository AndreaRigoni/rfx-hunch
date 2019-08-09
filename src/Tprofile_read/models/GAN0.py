
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

from models.base import GAN
from models.base import Dataset

# The reason we don't use bias is that we use BatchNormalization, which counteracts the role of bias.
# The role of bias:
# The network fitting ability is improved and the calculation is simple (only one addition).
# The improvement of capability comes from adjusting the overall distribution of output.
# Batchnormalization already includes the addition of the bias term. Recap that BatchNorm is already:
# gamma * normalized(x) + bias
# So there is no need (and it makes no sense) to add another bias term in the convolution layer. 
# Simply speaking BatchNorm shifts the activation by their mean values. Hence, any constant will be canceled out.


"""
With the training of the neural network, the input distribution of the network layer will change and gradually approach the two ends of the activation function, such as sigmoid activation function.
At this time, it will enter a saturated state, the gradient updates slowly, is insensitive to input changes, and even the gradient disappears, which makes it difficult to train the model.
BN, which is added before the input value of the activation function is input in the network layer, can pull the distribution to the normal distribution with the mean value of 0 and the standard deviation of 1.
The activation function is located in the region sensitive to the input value, thus accelerating the training of the model. In addition, BN can also play a regular role similar to dropout, because we will have
"Forced" operation, so the initialization requirements are not so high, you can use a larger learning rate.
"""
# model.add(tf.keras.layers.BatchNormalization())
"""
When the input of relu activation function is negative, the activation value is 0. At this time, the neuron can not learn.
When the input of leakyrelu activation function is negative, the activation value is not 0 (but the value is very small), and the neurons can continue to learn.
"""
# model.add(tf.keras.layers.LeakyReLU())


# def make_generator_model():
#     model = tf.keras.Sequential()
#     model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())
#     model.add(layers.Reshape((7, 7, 256)))
#     assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size
#     model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
#     assert model.output_shape == (None, 7, 7, 128)
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())
#     model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
#     assert model.output_shape == (None, 14, 14, 64)
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())
#     model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
#     assert model.output_shape == (None, 28, 28, 1)
#     return model
# ￼￼
# def make_discriminator_model():
#     model = tf.keras.Sequential()
#     model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
#                                      input_shape=[28, 28, 1]))
#     model.add(layers.LeakyReLU())
#     model.add(layers.Dropout(0.3))

#     model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
#     model.add(layers.LeakyReLU())
#     model.add(layers.Dropout(0.3))

#     model.add(layers.Flatten())
#     model.add(layers.Dense(1))

#     return model




# def train(dataset, epochs):
#   for epoch in range(epochs):
#     for image_batch in dataset:
#       train_step(image_batch)




"""
..######......###....##....##
.##....##....##.##...###...##
.##.........##...##..####..##
.##...####.##.....##.##.##.##
.##....##..#########.##..####
.##....##..##.....##.##...###
..######...##.....##.##....##
"""

class GAN0(GAN):

    def __init__(self, feature_dim=40, latent_dim=2, dprate = 0., scale=1, activation=tf.nn.relu, beta=1.):
        super(GAN0, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.dprate = dprate
        self.scale = scale
        self.activation = activation
        self.beta = beta

        self.set_model()
        self.set_optimizers()
        print('GAN0 ready:')


    def set_optimizers(self, opt=tf.keras.optimizers.Adam, dfactor=1e-3):
        # self.optimizers = [
        #     opt(dfactor),
        #     opt(dfactor)
        # ]
        self.gen_optimizer = tf.keras.optimizers.Adam(dfactor)
        self.dis_optimizer = tf.keras.optimizers.Adam(dfactor)

    def set_model(self, training=True):
        feature_dim = self.feature_dim
        latent_dim = self.latent_dim
        if training: dprate = self.dprate
        else: dprate = 0.
        scale = self.scale
        # activation = self.activation
        
        ## INFERENCE ##
        self.inference_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(feature_dim)),
            tf.keras.layers.Reshape( target_shape=(2,int(feature_dim/2),1) ),
            tf.keras.layers.Conv2D(filters=32*scale, kernel_size=(2,3), strides=(1, 1), padding='SAME'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(dprate),            
            # tf.keras.layers.Conv2D(filters=32*scale, kernel_size=(2,3), strides=(1, 1), padding='SAME'),
            # tf.keras.layers.LeakyReLU(),
            # tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(feature_dim * 20 * scale),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(feature_dim * 10 * scale),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(1),
        ] )

        # # ## GENERATION ##
        self.generative_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(latent_dim,)),            
            tf.keras.layers.Dense(latent_dim),
            tf.keras.layers.Dense(feature_dim * 10 * scale, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(feature_dim * 20 * scale, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape(target_shape=(2, int(feature_dim/2), int(20*scale) )),
            tf.keras.layers.Conv2DTranspose(filters=32*scale, kernel_size=(2,3), strides=(1, 1), use_bias=False, padding="SAME"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            # tf.keras.layers.Conv2DTranspose(filters=32*scale, kernel_size=(2,3), strides=(1, 1), use_bias=False, padding="SAME"),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(1,1), strides=(1, 1), padding="SAME"),
            tf.keras.layers.Flatten(),
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

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self, data, training=True):
        # generator_optimizer = self.optimizers[0]
        # discriminator_optimizer = self.optimizers[1]
        batch_size = tf.shape(data)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])
        # noise = tf.random.normal([batch_size, self.latent_dim], mean=0.5, stddev=0.5)
        # noise = tf.clip_by_value(noise,0.,1.)
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        def discriminator_loss(real_output, fake_output):
            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            total_loss = real_loss + fake_loss
            return total_loss

        def generator_loss(fake_output):        
            return cross_entropy(tf.ones_like(fake_output), fake_output)
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.generative_net(noise, training=True)
            real_output = self.discriminate(data, training=True)
            fake_output = self.discriminate(generated_data, training=True)
            gen_loss  = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        if training:
            gradients_of_generator = gen_tape.gradient(gen_loss, self.generative_net.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.inference_net.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generative_net.trainable_variables))
            self.dis_optimizer.apply_gradients(zip(gradients_of_discriminator, self.inference_net.trainable_variables))
        
        return tf.reduce_sum([gen_loss, disc_loss])

    

    def train(self, dataset, epoch=10, batch=200, optimizer_factor=1e-3):
        from tqdm import tqdm
        self.set_optimizers(dfactor=optimizer_factor)
        dataset_test  = dataset[0:batch]
        dataset_train = dataset[batch+1:]        
        ds = dataset_test.ds_array
        dt = dataset_train.ds_array
        def print_test_loss():
            ts_loss = 0
            for data in ds.batch(batch):
                x,_ = data
                ts_loss = self.train_step(x, training=False)
            print("test loss: ",ts_loss.numpy())

        for e in range(epoch):
            import sys
            print('EPOCH: ',e)
            # for i,data in enumerate(tqdm(dt.batch(batch), total=len(dataset_train)/batch+1)):
            for i,data in enumerate(dt.batch(batch)):
                x,_ = data
                loss = self.train_step(x)                
                sys.stdout.write('it: %d - loss: %f \r'%(i,loss))
                if self.stop_training:
                    break
            print_test_loss()
            if self.stop_training:
                print(' Train early stopping ... ')
                break

                
            







