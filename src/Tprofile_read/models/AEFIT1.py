


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




def tf_nan_to_num(x, num=0.):
    return tf.where(tf.math.is_nan(x), tf.ones_like(x) * num, x)


"""
.##....##....###....##....##....########..########.##....##..######..########
.###...##...##.##...###...##....##.....##.##.......###...##.##....##.##......
.####..##..##...##..####..##....##.....##.##.......####..##.##.......##......
.##.##.##.##.....##.##.##.##....##.....##.######...##.##.##..######..######..
.##..####.#########.##..####....##.....##.##.......##..####.......##.##......
.##...###.##.....##.##...###....##.....##.##.......##...###.##....##.##......
.##....##.##.....##.##....##....########..########.##....##..######..########
"""


from tensorflow.keras import activations
from tensorflow.keras import applications
from tensorflow.keras import backend
from tensorflow.keras import callbacks
from tensorflow.keras import constraints
from tensorflow.keras import datasets
from tensorflow.keras import estimator
from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import preprocessing
from tensorflow.keras import regularizers
from tensorflow.keras import utils    

class NaNDense2(tf.keras.layers.Dense):
  """Just your regular densely-connected NN layer.

  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.

  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.

  Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, units)`.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)
    super(NaNDense2, self).__init__(activity_regularizer=regularizers.get(activity_regularizer), **kwargs)


  def call(self, inputs):
    inputs = tf.convert_to_tensor(inputs)
    inputs = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
    
    outputs = tf.matmul(inputs, self.kernel)

    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      return self.activation(outputs)  # pylint: disable=not-callable
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if input_shape[-1] is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(NaNDense, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))











"""
.##.....##..#######..########..########.##......
.###...###.##.....##.##.....##.##.......##......
.####.####.##.....##.##.....##.##.......##......
.##.###.##.##.....##.##.....##.######...##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##..#######..########..########.########
"""

class AEFIT(tf.keras.Model):
    ''' General Autoencoder Fit Model for TF 2.0
    '''
    
    def __init__(self, feature_dim=40, latent_dim=2, latent_intervals=None):
        super(AEFIT, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.dprate = 0.2
        self.set_model()
        print('aefit 0 configured')

    def set_model(self, training=True):
        feature_dim = self.feature_dim
        latent_dim = self.latent_dim
        if training: dprate = self.dprate
        else: dprate = 0.
        self.nan_mask = NaNMask(feature_dim)
        ## INFERENCE ##
        self.inference_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(feature_dim,)),
            NaNDense(feature_dim, activation=tf.nn.relu),  #, activity_regularizer=tf.keras.regularizers.l1_l2(0.001)
            # tf.keras.layers.Dense(feature_dim, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(latent_dim * 200, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(latent_dim * 200, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(latent_dim * 100, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(2*latent_dim),
            ] )
        ## GENERATION ##
        self.generative_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(units=latent_dim, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(latent_dim * 100, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(latent_dim * 200, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(latent_dim * 200, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dprate),
            tf.keras.layers.Dense(units=feature_dim),
        ] )
        self.inference_net.build()
        self.generative_net.build()

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=([-1,self.latent_dim]))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, X):
        # X = self.nan_mask(X)
        X = tf.clip_by_value(X,0.,1.)
        mean, logvar = tf.split(self.inference_net(X), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, s, apply_sigmoid=False):
        x = self.generative_net(s)
        if apply_sigmoid:
            x = tf.sigmoid(x)
        return x

    def compute_loss(self, input):
        def vae_logN_pdf(sample, mean, logvar, raxis=1):
            log2pi = tf.math.log(2. * np.pi)
            return tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
        xy = tf_nan_to_num(input)        
        mean,logv = self.encode(xy)
        z = self.reparameterize(mean,logv)
        XY = tf_nan_to_num(self.decode(z))

        #
        crossen =  tf.nn.sigmoid_cross_entropy_with_logits(logits=XY, labels=xy)
        logpx_z = -tf.reduce_sum(crossen, axis=[1])
        logpz   =  vae_logN_pdf(z, 0., 1.)
        logqz_x =  vae_logN_pdf(z, mean, logv)
        l_vae   = -tf.reduce_mean(logpx_z + logpz - logqz_x)
        #   
        return l_vae

    def plot_generative(self, z):
        s = self.decode(tf.convert_to_tensor([z]),apply_sigmoid=True) 
        x,y = tf.split(s,2,axis=1)
        plt.plot(x[0],y[0])

    def save(self, filename):
        self.inference_net.save_weights(filename+'_encoder.kcp')
        self.generative_net.save_weights(filename+'_decoder.kcp')

    def load(self, filename):
        self.inference_net.load_weights(filename+'_encoder.kcp')
        self.generative_net.load_weights(filename+'_decoder.kcp')
        


def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = model.compute_loss(x)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))

def vae_log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)






def test_dummy(model, data, counts=60000, epoch=40, batch=400, loss_factor=1e-3):
    import seaborn as sns
    import Dummy_g1data as g1
    from sklearn.cluster import KMeans
    
    fig = plt.figure('aefit_test')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # ts,ls = data.ds_array.shuffle(counts).batch(batch).make_one_shot_iterator().get_next()

    count = 0
    optimizer = tf.keras.optimizers.Adam(loss_factor)
    g1.test_gendata(data)
    for e in range(epoch):
        ds = data.ds_array.shuffle(counts).batch(batch)
        for X in ds:
                X_data,_ = X                
                gradients, loss = compute_gradients(model, X_data)
                apply_gradients(optimizer, gradients, model.trainable_variables)
                count += 1

                if count % 20 == 0:
                  print('%d-%d loss: %f'%(e,count,tf.reduce_mean(loss)))                    
                  
                  m,v = model.encode(X_data)
                  # m,v = model.encode(ts)
                  z   = model.reparameterize(m,v)
                  XY  = model.decode(z,apply_sigmoid=True)
                  X,Y = tf.split(XY,2, axis=1)
                  
                  ax1.clear()
                  ax1.plot(m[:,0],m[:,1],'.')
                  # plt.plot(v[:,0],v[:,1],'.')
                  
                  ax2.clear()
                  # for i in range(model.latent_dim):
                  for i in range(batch):
                      # sns.scatterplot(X[i],Y[i])
                      ax2.plot(X[i],Y[i],'.')
              #     # plt.figure(2)
              #     # plt.clf()                
              #     # plt.plot(model._x,s[0])
              #     # sns.scatterplot(x[0],y[0])
                  fig.canvas.draw()
                  # plt.pause(0.1)




def plot_supervised_latent_distributions(model, counts=10000):
    import Dummy_g1data as g1
    dc = g1.Dummy_g1data(counts=counts, size=int(model.feature_dim/2), noise_var=0.)
    data = dc.ds_array.batch(1)
    clist=['red','blue','green','purple','grey']
    plt.figure('plot_supervised_latent_distributions')
    plt.clf()
    count = 0
    for X in data:
        ds,dl = X
        m,v = model.encode(ds)
        z   = model.reparameterize(m,v)
        #XY  = model.decode(z,apply_sigmoid=True)
        plt.plot(z[:,0],z[:,1],'.',color=clist[dl%len(clist)])
        count += 1
        if count % 100 == 0:
            plt.pause(0.001)
    