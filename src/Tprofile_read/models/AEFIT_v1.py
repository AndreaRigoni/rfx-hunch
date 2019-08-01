

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

"""
.##.....##..#######..########..########.##......
.###...###.##.....##.##.....##.##.......##......
.####.####.##.....##.##.....##.##.......##......
.##.###.##.##.....##.##.....##.######...##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##..#######..########..########.########
"""




class AEFIT_v1(VAE):
    ''' General Autoencoder Fit Model for TF 1.x
    '''

    def __init__(self, feature_dim=40, latent_dim=2, latent_intervals=None):
        super(AEFIT_v1, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.set_model()
        
    def set_model(self):
        feature_dim = self.feature_dim
        latent_dim = self.latent_dim
        ## INFERENCE ##
        self.inference_net = tf.keras.Sequential(
            [
            # tf.keras.layers.Input(shape=(feature_dim,)),
            tf.keras.layers.Dense(feature_dim, activation=tf.nn.relu, input_shape=(feature_dim,)),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(latent_dim * 200, activation=tf.nn.relu),
            tf.keras.layers.Dense(latent_dim * 100, activation=tf.nn.relu),
            tf.keras.layers.Dense(2*latent_dim),
            ]
        )
        ## GENERATION ##
        self.generative_net = tf.keras.Sequential(
        [
            # tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(units=latent_dim, activation=tf.nn.relu, input_shape=(latent_dim,)),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(latent_dim * 100, activation=tf.nn.relu),
            tf.keras.layers.Dense(latent_dim * 200, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=feature_dim),
        ]
        )
        self.inference_net.build()
        self.generative_net.build()


    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=([-1,self.latent_dim]))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, X):
        X = tf.clip_by_value(X,0.,1.)
        mean, logvar = tf.split(self.inference_net(X), num_or_size_splits=2, axis=1)
        return mean, logvar        

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=(self.latent_dim,))
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
        xy = input
        mean,logv = self.encode(xy)
        z = self.reparameterize(mean,logv)
        XY = self.decode(z)
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
        



def test_dummy(model, data, epoch=10, batch=400, learning_rate=1e-3):    
    import seaborn as sns
    import Dummy_g1data as g1
    from sklearn.cluster import KMeans

    def compute_gradients(model, x):
        with tf.GradientTape() as tape:
            loss = model.compute_loss(x)
        return tape.gradient(loss, model.trainable_variables), loss

    def apply_gradients(optimizer, gradients, variables):
        return optimizer.apply_gradients(zip(gradients, variables))
    
    fig = plt.figure('aefit_test')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.tight_layout()
    
    ## TRAIN
    ## 
    ds = data.ds_array.batch(batch).prefetch(batch)
    it = ds.make_initializable_iterator()
    X,_ = it.get_next()
    loss = model.compute_loss(X)    
    
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # train_op = optimizer.minimize(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients, loss = compute_gradients(model, X)
    train_op = apply_gradients(optimizer, gradients, model.trainable_variables)    

    ## TEST
    ##
    it_test = data.ds_array.batch(batch).make_initializable_iterator()
    
    count = 0
    # with tf.Session() as sess:
    sess = tf.get_default_session()
    sess.run( tf.global_variables_initializer() )
    sess.run( it_test.initializer )
    ts,_ = sess.run(it_test.get_next())
    for e in range(epoch):            
        sess.run(it.initializer)
        while True:
            count += 1
            try: sess.run(train_op)
            except tf.errors.OutOfRangeError: break
            if count % 20 == 0:
                acc = sess.run(model.compute_loss(ts))
                print('%d-%d loss: %f'%(e,count,acc))
                x,y = sess.run(tf.split(ts,2, axis=1))
                m,v = sess.run(model.encode(ts))
                z   = model.reparameterize(m,v)
                XY  = model.decode(z,apply_sigmoid=True)
                X,Y = sess.run(tf.split(XY,2, axis=1))
                
                ax1.clear()
                ax1.plot(m[:,0],m[:,1],'.')
                # # ax1.plot(v[:,0],v[:,1],'.')
                # #                         
                ax2.clear()                    
                for i in range(batch):
                    # ax2.plot(x[i],y[i],'.')
                    ax2.plot(X[i],Y[i],'.')
                fig.canvas.draw()




