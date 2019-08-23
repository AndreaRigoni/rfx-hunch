
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
        return self
    
    def compose(self, autoencoders):
        from itertools import chain 
        # for m in vaes: assert issubclass(type(m), VAE), 'please set a valid VAE as inner model'
        vaes = autoencoders
        i_inputs   = list(chain.from_iterable([ m.inference_net.inputs for m in vaes ]))
        i_outputs  = list(chain.from_iterable([ m.inference_net.outputs for m in vaes ]))
        inf_split  = tf.keras.layers.Lambda( lambda x: tf.split(x, num_or_size_splits=2, axis=1)[0] )
        inf_append = tf.keras.layers.Concatenate()( [ inf_split(x) for x in i_outputs ] )
        self.inference_net = tf.keras.Model(i_inputs,[self._model.inference_net(inf_append)], name='compose_inference_net')

        g_outputs = list(chain.from_iterable([ m.generative_net.outputs for m in vaes ]))
        g_inputs  = list(chain.from_iterable([ m.generative_net.inputs for m in vaes ]))
        gen_append = tf.keras.Model(g_inputs, g_outputs)
        splits_z = [ inpt.shape[1] for inpt in g_inputs ]        
        dout  = tf.keras.layers.Dense(sum(splits_z))(self._model.generative_net.output)
        gen_split = tf.keras.layers.Lambda( lambda x: tf.split(x,num_or_size_splits=splits_z, axis=1))(dout)        
        self.generative_net = tf.keras.Model(self._model.generative_net.inputs, [gen_append(gen_split)], name='compose_generative_net')
        
        i_inputs_shapes = [ m.inference_net.input_shape for m in vaes ]
        print(i_inputs_shapes)
        self.build(input_shape=i_inputs_shapes)

        self._mkids = vaes
        self.compile(
            optimizer= tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics  = ['accuracy']
        )
        return self

    def call(self, xy, training=None):
        mean, logvar = tf.split(self.inference_net(xy, training=training), num_or_size_splits=2, axis=1)
        z  = self._model.reparametrize(mean, logvar)
        XY = self.generative_net(z, training=training)
        return XY

    def compile(self, optimizer, loss=None, metrics=None, **kwargs):
        loss_list = [ m.loss for m in self._mkids ]
        return self.super.compile(optimizer, loss=loss_list, metrics=metrics, **kwargs)




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
        if loss is not None: print('WARNING: you are tryig to set a loss where losses comes directly from compose models')        
        loss_list = [ m.loss for m in self._mkids ]
        return self.super.compile(optimizer, loss=loss_list, metrics=metrics, **kwargs)


# class Compose2(models.base.VAE):

#     def __init__(self, *args, **kwargs):
#         self.super = super(Compose2, self)
#         self.super.__init__()
#         self._model = None # instance of model among inner latent sapaces
#         self._mkids = None # all childs models to be composed

#     def set_model(self, model):
#         self._model = model
#         return self

#     def compose(self, autoencoders):
#         from itertools import chain         
#         vaes = autoencoders
        
#         i_inputs   = list(chain.from_iterable([ m.inference_net.inputs for m in vaes ]))
#         i_outputs  = [ m.rep.output for m in vaes ]
#         g_inputs    = list(chain.from_iterable([ m.generative_net.inputs for m in vaes ]))                
#         g_outputs   = list(chain.from_iterable([ m.generative_net.outputs for m in vaes ]))
        
#         concat     = tf.keras.layers.Concatenate()(i_outputs)
#         compose_out = self._model(concat)
#         splits_siz  = [ inpt.shape[1] for inpt in g_inputs ]
#         gen_split   = tf.keras.layers.Lambda( lambda x: tf.split(x,num_or_size_splits=splits_siz, axis=1))(compose_out)
#         gen_append  = tf.keras.Model(g_inputs, g_outputs)
#         gen_outs    = gen_append(gen_split)

#         self.super.__init__(i_inputs, gen_outs)
#         self.build(input_shape=[ m.inference_net.input_shape for m in vaes ])
        
#         self._mkids = vaes
#         self.compile(
#             optimizer= tf.keras.optimizers.Adam(learning_rate=1e-3),
#             metrics  = ['accuracy']
#         )
#         return self

#     def call(self, xy, training=None):
#         return self.super.call(xy, training=training)

#     def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
#         if optimizer is None: 
#             if self.optimizer: optimizer = self.optimizer
#             else             : optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#         if loss is not None: print('WARNING: you are tryig to set a loss where losses comes directly from compose models')        
#         loss_list = [ m.loss for m in self._mkids ]
#         return self.super.compile(optimizer, loss=loss_list, metrics=metrics, **kwargs)

