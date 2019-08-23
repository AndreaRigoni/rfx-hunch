


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
            #    use_bias=True,
            #    kernel_initializer='glorot_uniform',
            #    bias_initializer='zeros',
            #    kernel_regularizer=None,
            #    bias_regularizer=None,
            #    activity_regularizer=None,
            #    kernel_constraint=None,
            #    bias_constraint=None,
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






class ResetCallback(tf.keras.callbacks.Callback):
  def on_epoch_begin(self, batch, logs=None):
    self.model.losses = []

"""
.##.....##..#######..########..########.##......
.###...###.##.....##.##.....##.##.......##......
.####.####.##.....##.##.....##.##.......##......
.##.###.##.##.....##.##.....##.######...##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##..#######..########..########.########
"""
class AEFIT4(models.base.VAE):
    ''' General Autoencoder Fit Model for TF 2.0
    '''    
    def __init__(self, feature_dim=40, latent_dim=2, dprate = 0., activation=tf.nn.relu, beta=1., 
                 encoder_geometry=[20,20,10], decoder_geometry=None, scale=1, *args, **kwargs):
                 super(AEFIT4, self).__init__(*args, **kwargs)
                 self.latent_dim = latent_dim
                 self.feature_dim = feature_dim
                 self.encoder_geometry = encoder_geometry
                 if decoder_geometry is None: decoder_geometry = encoder_geometry[::-1] # reverse order
                 self.decoder_geometry = decoder_geometry
                 self.dprate = dprate
                 self.scale = scale
                 self.activation = activation
                 self.beta = beta        
                 self.set_model(feature_dim, latent_dim, *args, **kwargs)
                 print('AEFIT4 ready:')
    

    def set_model(self, feature_dim, latent_dim, dprate=0., scale=1., activation='relu', *args, **kwargs):
        def add_dense_encode(self, fdim=feature_dim, ldim=latent_dim, geometry=[20,20,10,10]):
            for i,size in enumerate(geometry):
                self.add(tf.keras.layers.Dense(fdim*size*scale, name=self.name+'_decode_'+str(i), activation=activation))
                self.add(tf.keras.layers.Dropout(dprate, name=self.name+'_dpout_'+str(i)))
            self.add(tf.keras.layers.Dense(ldim))
            return self

        def add_dense_decode(self, fdim=feature_dim, ldim=latent_dim, geometry=[10,10,20,20]):            
            for i,size in enumerate(geometry):
                self.add(tf.keras.layers.Dense(fdim*size*scale, name=self.name+'_decode_'+str(i), activation=activation))
                self.add(tf.keras.layers.Dropout(dprate, name=self.name+'_dpout_'+str(i)))
            self.add(tf.keras.layers.Dense(fdim))
            return self
        # add methods to Sequential class
        tf.keras.Sequential.add_dense_encode = add_dense_encode
        tf.keras.Sequential.add_dense_decode = add_dense_decode
        
        ## INFERENCE ##
        inference_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(feature_dim,)),
            NaNDense(feature_dim),
        ], name=self.name+'_seq_inference_net' ).add_dense_encode(ldim=2*latent_dim, geometry=self.encoder_geometry)

        ## GENERATION ##
        generative_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(latent_dim)
        ], name=self.name+'_seq_generative_net' ).add_dense_decode(geometry=self.decoder_geometry)

        inference_net.build()
        generative_net.build()
        self.inference_net = inference_net
        self.generative_net = generative_net

        self._sce = 0.
        self._mkl = 0.
        self._akl = 0.        
        # Compile the model
        self.compile(  
            optimizer= tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss     = self.compute_loss,
            metrics  = ['accuracy', self.sce, self.akl, self.mkl]
        )    


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
        if apply_sigmoid: x = tf.sigmoid(x)
        return x

    def call(self, xy, training=None):
        
        @tf.function
        def analytic_kld(mean, logvar, axis=1):
            return -0.5 * tf.reduce_sum(1. + logvar - tf.square(mean) - tf.exp(logvar)  , axis=axis)
         

        @tf.function
        def montecarlo_kld(z, mean, logvar, axis=1):
            def vae_logN_pdf(sample, mean, logvar):
                log2pi = tf.math.log(2. * np.pi)
                return -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)
            logpz   =  vae_logN_pdf(z, 0., 1.)
            logqz_x =  vae_logN_pdf(z, mean, logvar)
            return    -tf.reduce_sum(logpz - logqz_x, axis=axis)

        att = tf.math.is_nan(xy)
        xy  = tf.where(att, tf.zeros_like(xy), xy)
        mean, logvar = self.encode(xy)
        z   = self.reparametrize(mean, logvar)        
        XY  = self.decode(z)
        XY  = tf.where(att, tf.zeros_like(XY), XY)

        # MONTECARLO KL LOSS #
        mkl_loss = montecarlo_kld(z,mean,logvar)
        
        # ANALYTIC GAUSSIAN KL TERM #
        akl_loss  = analytic_kld(mean,logvar)
        akl_weight = tf.reduce_sum(tf.dtypes.cast( tf.math.is_nan(xy), dtype=tf.float32 ), axis=1)
        akl_loss  += 0.5 * akl_weight
        self.add_loss( lambda: self.beta * akl_loss )

        # DEBUG
        self._mkl =  tf.reduce_max(mkl_loss)
        self._akl =  tf.reduce_max(akl_loss)

        return XY

    def compute_loss(self, xy, XY):
        xy = tf.where(tf.math.is_nan(xy), tf.zeros_like(xy), xy)
        crossen =  tf.nn.sigmoid_cross_entropy_with_logits(logits=XY, labels=xy)
        logpx_z =  tf.reduce_sum( crossen , axis=1)
        return logpx_z


    ## IF SEEMS TO FAIL TO FIND A GOOD CONVERGENCE
    def train_step(self, xy, training=True):
        with tf.GradientTape() as tape:
            XY = self.call(xy, training=training)
            loss = tf.reduce_mean(self.compute_loss(xy, XY) + tf.convert_to_tensor(self.losses))
        if training:
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def recover(self,x):
        xr = self.call(x, training=False)
        return tf.where(tf.math.is_nan(x),xr,x)

    # KL divergence by montecarlo
    def mkl(self, x, y):
        return self._mkl

    # analytical Kullback Libler
    def akl(self, x, y):
        return self._akl

    # sigmoid cross entropy
    def sce(self, x, y):
        return self._sce

    def plot_generative(self, z):
        s = self.decode(tf.convert_to_tensor([z]),apply_sigmoid=True) 
        x,y = tf.split(s,2,axis=1)
        plt.plot(x[0],y[0])        






    