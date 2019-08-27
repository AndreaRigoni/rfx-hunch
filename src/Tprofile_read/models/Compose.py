
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import abc

import models.base

class Compose(models.base.VAE):

    def __init__(self, *args, **kwargs):
        self.super = super(Compose, self)
        self.super.__init__()
        self._model = None # instance of model among inner latent sapaces
        self._mkids = None # all childs models to compose
        pass

    def set_model(self, model):
        self._model = model
        self.latent_dim = model.latent_dim
        return self
    
    def compose(self, autoencoders):
        from itertools import chain 
        # for m in vaes: assert issubclass(type(m), VAE), 'please set a valid VAE as inner model'
        vaes = autoencoders
        def get_inference_inputs(m):
            if hasattr(m, 'inference_net'): return m.inference_net.inputs
            else                          : return m.inputs
        def get_inference_outputs(m):
            if hasattr(m, 'inference_net'): 
                out = m.inference_net.outputs
                inf_split = tf.keras.layers.Lambda( lambda x: tf.split(x, num_or_size_splits=2, axis=1)[0] )
                return [ inf_split(x) for x in out ]
            else: 
                return m.outputs
        def get_generative_inputs(m):
            if hasattr(m, 'generative_net'): return m.generative_net.inputs
            else                          : return m.inputs
        def get_generative_outputs(m):
            if hasattr(m, 'generative_net'): return m.generative_net.outputs
            else                          : return m.outputs
        def get_inference_inputshape(m):
            if hasattr(m, 'inference_net'): return m.inference_net.input_shape
            else                          : return m.input_shape


        i_inputs   = list(chain.from_iterable([ get_inference_inputs(m) for m in vaes ]))
        i_outputs  = list(chain.from_iterable([ get_inference_outputs(m) for m in vaes ]))
        inf_append = tf.keras.layers.Concatenate()( i_outputs )
        self.inference_net = tf.keras.Model(i_inputs, self._model.inference_net(inf_append), name='compose_inference_net')
        
        g_outputs = list(chain.from_iterable([ get_generative_outputs(m) for m in vaes ]))
        g_inputs  = list(chain.from_iterable([ get_generative_inputs(m) for m in vaes ]))
        gen_append = tf.keras.Model(g_inputs, g_outputs)
        
        splits_z = [ inpt.shape[1] for inpt in g_inputs ]

        dout  = self._model.generative_net.output
        gen_split = tf.keras.layers.Lambda( lambda x: tf.split(x,num_or_size_splits=splits_z, axis=1))(dout)        
        self.generative_net = tf.keras.Model(self._model.generative_net.inputs, gen_append(gen_split), name='compose_generative_net')        
        
        i_inputs_shapes = [ get_inference_inputshape(m) for m in vaes ]


        print(i_inputs_shapes)
        # self.build(input_shape=i_inputs_shapes)
        self.inference_net.build(input_shape=i_inputs_shapes)
        self.generative_net.build(input_shape=self._model.generative_net.input_shape)

        self._mkids = vaes
        self.compile(
            optimizer= tf.keras.optimizers.Adam(learning_rate=1e-3),            
            metrics  = ['accuracy']
        )
        return self

    @tf.function
    def reparametrize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    @tf.function
    def encode(self, X, training=False):
        mean, logvar = tf.split(self.inference_net(X, training=training), num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def decode(self, s, training=False, apply_sigmoid=None):
        x = self.generative_net(s, training=training)
        if apply_sigmoid is True:
            if isinstance(x, list): x = [ tf.sigmoid(X) for X in x]
        return x
    
    def call(self, xy, training=True):
        mean, logvar = tf.split(self.inference_net(xy, training=training), num_or_size_splits=2, axis=1)        
        # z  = tf.cond(training, lambda: self._model.reparametrize(mean, logvar), lambda: mean )
        z  = self._model.reparametrize(mean, logvar)
        XY = self.generative_net(z, training=training)
        return XY

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        if optimizer is None: 
            if self.optimizer: optimizer = self.optimizer
            else             : optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        if loss is not None: 
            print('WARNING: you are tryig to set a loss where losses should come directly from compose models')
        else: loss = [ m.loss for m in self._mkids ]
        return self.super.compile(optimizer, loss=loss, metrics=metrics, **kwargs)






class Compose1(models.base.VAE):

    def __init__(self, *args, **kwargs):
        self.super = super(Compose1, self)
        self.super.__init__()
        self._model = None # instance of model among inner latent sapaces
        self._mkids = None # all childs models to be composed

    def set_model(self, model):
        self._model = model
        return self

    def compose(self, autoencoders):
        from itertools import chain         
        vaes = autoencoders
        
        i_inputs   = list(chain.from_iterable([ m.inference_net.inputs for m in vaes ]))
        i_outputs  = list(chain.from_iterable([ m.inference_net.outputs for m in vaes ]))        
        inf_append = tf.keras.Model(i_inputs, i_outputs)
        inf_split  = tf.keras.layers.Lambda( lambda x: tf.split(x, num_or_size_splits=2, axis=1)[0] )

        if len(vaes) > 1: inf_conct = tf.keras.layers.Concatenate()( [ inf_split(x) for x in inf_append.outputs ] )
        else            : inf_conct = inf_append.outputs
        g_outputs   = list(chain.from_iterable([ m.generative_net.outputs for m in vaes ]))
        g_inputs    = list(chain.from_iterable([ m.generative_net.inputs for m in vaes ]))                
        gen_append  = tf.keras.Model(g_inputs, g_outputs)
        splits_siz  = [ inpt.shape[1] for inpt in g_inputs ]
        compose_out = self._model(inf_conct)
        gen_split   = tf.keras.layers.Lambda( lambda x: tf.split(x,num_or_size_splits=splits_siz, axis=1))(compose_out)
        gen_outs    = gen_append(gen_split)

        self.super.__init__(i_inputs, gen_outs)
        self.build(input_shape=[ m.inference_net.input_shape for m in vaes ])
        self._mkids = vaes
        self.compile(
            optimizer= tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics  = ['accuracy']
        )
        return self


    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        if optimizer is None: 
            if self.optimizer: optimizer = self.optimizer
            else             : optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        if loss is None: loss = [ m.loss for m in self._mkids ]
        else           : loss = loss([ m.loss for m in self._mkids ])
        return self.super.compile(optimizer, loss=loss, metrics=metrics, **kwargs)


