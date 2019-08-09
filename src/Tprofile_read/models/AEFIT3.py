


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

class AEFIT3(models.base.VAE):
    ''' General Autoencoder Fit Model for TF 2.0
    '''
    
    def __init__(self, feature_dim=40, latent_dim=2, dprate = 0., scale=1, activation=tf.nn.relu, beta=1.):
        super(AEFIT3, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.dprate = dprate
        self.scale = scale
        self.activation = activation
        self.set_model()
        self.beta = beta
        print('AEFIT3 pz ready:')

    def set_model(self, training=True):
        feature_dim = self.feature_dim
        latent_dim = self.latent_dim
        if training: dprate = self.dprate
        else: dprate = 0.
        scale = self.scale
        activation = self.activation
        
        ## INFERENCE ##
        self.inference_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(feature_dim,)),
            NaNDense(feature_dim, activation=activation),
            # tf.keras.layers.Dense(feature_dim, activation=activation),
            tf.keras.layers.Dense(latent_dim * 200 * scale, activation=activation),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(latent_dim * 200 * scale, activation=activation),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(latent_dim * 100 * scale, activation=activation),            
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(latent_dim * 100 * scale, activation=activation),            
            tf.keras.layers.Dense(2*latent_dim),
            ] )

        ## GENERATION ##
        self.generative_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(units=latent_dim),
            tf.keras.layers.Dense(latent_dim * 100 * scale, activation=activation),            
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(latent_dim * 100 * scale, activation=activation),            
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(latent_dim * 200 * scale, activation=activation),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(latent_dim * 200 * scale, activation=activation),
            tf.keras.layers.Dense(units=feature_dim),
        ] )
        self.inference_net.build()
        self.generative_net.build()

        self.inputs = self.inference_net.inputs
        self.outputs = self.generative_net.outputs
        self._sce = 0.
        self._kld = 0.
        self._akl = 0.
        self._v_mea = 0.
        self._v_std = 0.
        # Compile the model
        self.compile(  
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss = self.vae3_loss,
            metrics = ['accuracy', self.sce, self.akl, self.kld, self.v_mea, self.v_std]
        )
        self.build(input_shape=self.inference_net.input_shape)

    loss_factor = property()
    
    @loss_factor.getter
    def loss_factor(self):
        return self.optimizer._get_hyper('learning_rate')

    @loss_factor.setter
    def loss_factor(self, lf):
        self.optimizer._set_hyper('learning_rate', lf)



    @tf.function
    def reparameterize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


    @tf.function
    def encode(self, X, training=True):
        X = tf.clip_by_value(X,0.,1.)
        mean, logvar = tf.split(self.inference_net(X, training=training), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def decode(self, s, training=True, apply_sigmoid=False):
        x = self.generative_net(s, training=training)
        if apply_sigmoid: x = tf.sigmoid(x)
        return x

    def call(self, xy):
        def vae_logN_pdf(sample, mean, logvar):
            log2pi = tf.math.log(2. * np.pi)
            return -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)

        def vae_N_pdf(z, mean, logvar):                        
            return tf.exp( -0.5 * (z - mean)**2 * tf.exp(-logvar) )

        att = tf.math.is_nan(xy)
        xy = tf.where(att, tf.zeros_like(xy), xy)
        mean,logvar = self.encode(xy)
        z = self.reparameterize(mean, logvar)        
        XY = self.decode(z)
        XY = tf.where(att, tf.zeros_like(XY), XY)
        
        # MONTECARLO KL LOSS #
        logpz   =  vae_logN_pdf(z, 0., 1.)
        logqz_x =  vae_logN_pdf(z, mean, logvar)
        kl_loss = -tf.reduce_sum(logpz - logqz_x, axis=1)
        
        # ANALYTIC GAUSSIAN KL TERM #
        akl_loss = -0.5 * tf.reduce_sum(logvar + 1 - tf.square(mean) + tf.exp(logvar)  , axis=1)

        # REGULARIZER FOR p(z) #
        m1,v1     = tf.roll(mean, 1, axis=0), tf.roll(logvar, 1, axis=0)
        pz        = tf.reduce_mean(vae_N_pdf(z,m1,v1), axis=1)
        akl_loss -= self.beta * pz

        # DEBUG
        crossen =  tf.nn.sigmoid_cross_entropy_with_logits(logits=XY, labels=xy)
        logpx_z =  tf.reduce_sum(crossen, axis=[1])        
        self._v_mea = tf.reduce_mean(mean)
        self._v_std = tf.reduce_mean(tf.exp(0.5 * logvar))
        self._sce =  tf.reduce_mean(logpx_z)
        self._kld =  tf.reduce_mean(kl_loss)
        self._akl =  tf.reduce_mean(akl_loss)
        
        self.add_loss(akl_loss)
        return XY
                
    def recover(self,x):
        xr = self.call(x)
        return tf.where(tf.math.is_nan(x),xr,x)

    # KL divergence by montecarlo
    def kld(self, x, y):
        return self._kld

    # analytical Kullback Libler
    def akl(self, x, y):
        return self._akl

    # sigmoid cross entropy
    def sce(self, x, y):
        return self._sce

    def v_mea(self, x, y):
        return self._v_mea
    def v_std(self, x, y):
        return self._v_std


    def vae3_loss(self, xy, XY):
        xy = tf.where(tf.math.is_nan(xy), tf.zeros_like(xy), xy)
        #XY = tf.where(tf.math.is_nan(XY), tf.zeros_like(XY), XY)
        crossen =  tf.nn.sigmoid_cross_entropy_with_logits(logits=XY, labels=xy)
        logpx_z =  tf.reduce_sum(crossen, axis=[1])
        l_term =   tf.convert_to_tensor(self.losses)
        l_vae   =  tf.reduce_mean(logpx_z - l_term )
        return l_vae        

    def plot_generative(self, z):
        s = self.decode(tf.convert_to_tensor([z]),apply_sigmoid=True) 
        x,y = tf.split(s,2,axis=1)
        plt.plot(x[0],y[0])        



