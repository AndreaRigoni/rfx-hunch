


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



class Reparametrize1D(tf.keras.layers.Layer):
    """ VAE REPARAMETRIZATION LAYER
    """
    def __init__(self, **kwargs):
        super(Reparametrize1D, self).__init__(**kwargs)
    
    @tf.function
    def reparametrize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    @tf.function
    def call(self, inputs, training=True):
        inputs = tf.convert_to_tensor(inputs)
        mean, logvar = tf.split(inputs, num_or_size_splits=2, axis=1)
        akl_loss = -0.5 * tf.reduce_sum(1. + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
        mean = self.reparametrize(mean,logvar)
        self.add_loss( akl_loss, inputs=True )
        return mean
        
    

class RelUnitNorm(tf.keras.constraints.Constraint):
    """Constrains the weights incident to each hidden unit to have unit norm.
    """
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, w):
        # w =  w / ( tf.keras.backend.epsilon() + 
        #      tf.sqrt( tf.reduce_sum(tf.square(w), axis=self.axis, keepdims=True)))
        w =  w / ( tf.keras.backend.epsilon() + 
             tf.sqrt( tf.reduce_max(tf.square(w), axis=self.axis, keepdims=True)))
        return w

    def get_config(self):
        return {'axis': self.axis}


tf.keras.Model.fit

class Relevance1D(tf.keras.layers.Dropout):
    def __init__(self,
                activation=None,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=None,
                kernel_constraint=RelUnitNorm(),
                **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Relevance1D, self).__init__( 0., **kwargs )
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)        
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

        self.supports_masking = True
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=1)

    def build(self, input_shape):
        dtype = tf.dtypes.as_dtype(self.dtype or tf.keras.backend.floatx)
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                            'should be defined. Found `None`.')
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=1, axes={-1: last_dim})
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,            
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
            
        self.built = True


    def call(self, inputs):
        inputs  = tf.convert_to_tensor(inputs)
        # inputs = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        outputs = tf.multiply( inputs , self.kernel )
        # if self.activation is not None:
        #     return self.activation(outputs)  # pylint: disable=not-callable
        outputs = super(Relevance1D, self).call(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError( 'The innermost dimension of input_shape must be defined, but saw: %s' % input_shape)
        return input_shape

    def get_config(self):
        config = {
            'activation': tf.keras.activations.serialize(self.activation),
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
        }
        base_config = super(Relevance1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



tf.keras.layers.Dropout


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
    def __init__(self, feature_dim=40, latent_dim=2, dprate=0., activation=tf.nn.relu, beta=1., 
                 geometry=[20,20,10], scale=1, *args, **kwargs):
        self.super = super(AEFIT5, self)
        self.super.__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim        
        self.dprate = dprate
        self.scale = scale
        self.activation = activation
        self.beta = beta 
        self.apply_sigmoid = False
        self.bypass = False
        
        inference_net, generative_net = self.set_model(feature_dim, latent_dim, 
                                                        dprate=dprate,
                                                        scale=scale, 
                                                        activation=activation,
                                                        geometry=geometry)
        self.inference_net = inference_net
        self.generative_net = generative_net        
        # self.build(input_shape=inference_net.input_shape)
        self.compile(
            optimizer  = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss       = self.compute_cross_entropy_loss,
            logit_loss = True,
            metrics    = ['accuracy']
        )
        print('AEFIT5 a ready:')


    
    def set_model(self, feature_dim, latent_dim, dprate=0., activation=tf.nn.relu, 
                  geometry=[20,20,10], scale=1):

        class LsInitializer(tf.keras.initializers.Initializer):
            """Initializer for latent layer"""
            def __init__(self, axis=1):
                super(LsInitializer, self).__init__()
                self.axis = axis

            def __call__(self, shape, dtype=tf.dtypes.float32):
                dtype = tf.dtypes.as_dtype(dtype)
                if not dtype.is_numpy_compatible or dtype == tf.dtypes.string:
                    raise ValueError("Expected numeric or boolean dtype, got %s." % dtype)
                axis = self.axis
                shape[axis] = int(shape[axis]/2)
                identity = tf.initializers.identity()(shape)
                return tf.concat([identity, tf.zeros(shape)], axis=axis)

        def add_dense_encode(self, fdim=feature_dim, ldim=latent_dim, geometry=[20,20,10,10]):
            for _,size in enumerate(geometry):
                self.add(tf.keras.layers.Dense(fdim*size*scale, activation=activation))
                self.add(tf.keras.layers.Dropout(dprate))
            if len(geometry) == 0: initializer = LsInitializer()
            else : initializer = None
            self.add(tf.keras.layers.Dense(ldim, activation='linear', use_bias=False, kernel_initializer=initializer))
            return self

        def add_dense_decode(self, fdim=feature_dim, ldim=latent_dim, geometry=[10,10,20,20]):            
            for _,size in enumerate(geometry):
                self.add(tf.keras.layers.Dense(fdim*size*scale, activation=activation))
                self.add(tf.keras.layers.Dropout(dprate))
            if len(geometry) == 0: initializer = tf.initializers.identity()
            else : initializer = None
            self.add(tf.keras.layers.Dense(fdim, activation='linear', use_bias=False, kernel_initializer=initializer))            
            return self
        # add methods to Sequential class
        tf.keras.Sequential.add_dense_encode = add_dense_encode
        tf.keras.Sequential.add_dense_decode = add_dense_decode
        
        ## INFERENCE ##
        inference_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(feature_dim,)),
            tf.keras.layers.Lambda(lambda x: tf.where(tf.math.is_nan(x),tf.zeros_like(x),x)), 
            # # NaNDense(feature_dim),
            Relevance1D(name=self.name+'_iRlv', activation='linear', kernel_initializer=tf.initializers.ones),
        ]).add_dense_encode(ldim=2*latent_dim, geometry=geometry)

        ## GENERATION ##
        generative_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(latent_dim,)),
            Relevance1D(name=self.name+'_gRlv', activation='linear', kernel_initializer=tf.initializers.ones),
            #tf.keras.layers.Dense(latent_dim)
        ]).add_dense_decode(geometry=geometry[::-1])
        
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
    def decode(self, s, training=True, apply_sigmoid=None):
        x = self.generative_net(s, training=training)
        if apply_sigmoid is None: apply_sigmoid = self.apply_sigmoid        
        if apply_sigmoid is True and training is False:
            x = tf.sigmoid(x)
        return x

    def call(self, xy, training=None):
        att = tf.math.is_nan(xy)
        xy  = tf.where(att, tf.zeros_like(xy), xy)
        mean, logvar = self.encode(xy, training=training)
        z = self.reparametrize(mean,logvar)
        XY = self.decode(z, training=training)        
        if training is not False:
            XY  = tf.where(att, tf.zeros_like(XY), XY)
        kl_loss = -0.5 * tf.reduce_sum(1. + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
        if self.bypass:
            XY = 0.*XY + xy # add dummy gradients passing through the ops
            kl_loss = 0.
        self.add_loss( lambda: self.beta * kl_loss, inputs=True )
        return XY
    
    ## IF SEEMS TO FAIL TO FIND A GOOD CONVERGENCE
    def train_step(self, data, training=True):
        xy = data[0]
        with tf.GradientTape() as tape:
            XY = self.call(xy, training=training)
            loss = tf.reduce_mean( self.loss(xy, XY) + self.losses[0] )
        if training:
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def compile(self, optimizer=None, loss=None, logit_loss=False, metrics=None, **kwargs):
        if optimizer is None: 
            if self.optimizer: optimizer = self.optimizer
            else             : optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
        if loss is None or (hasattr(self, 'loss') and loss == self.loss): loss_wrapper = self.loss
        else: 
            self.apply_sigmoid = logit_loss
            loss_wrapper = lambda xy,XY: loss( tf.where(tf.math.is_nan(xy), tf.zeros_like(xy), xy), 
                                               tf.where(tf.math.is_nan(xy), tf.zeros_like(XY), XY))
        return self.super.compile(optimizer, loss=loss_wrapper, metrics=metrics, **kwargs)

    def compute_cross_entropy_loss(self, xy, XY):
        crossen =  tf.nn.sigmoid_cross_entropy_with_logits(logits=XY, labels=xy)
        logpx_z =  tf.reduce_sum( crossen , axis=1)
        return logpx_z

    def recover(self,x):
        xr = self.call(x, training=False)
        return tf.where(tf.math.is_nan(x),xr,x)







    