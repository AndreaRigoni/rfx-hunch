

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
import Hunch_utils as Hunch


"""
..######...#######..##....##.##.....##.......##...########.
.##....##.##.....##.###...##.##.....##.....####...##.....##
.##.......##.....##.####..##.##.....##.......##...##.....##
.##.......##.....##.##.##.##.##.....##.......##...##.....##
.##.......##.....##.##..####..##...##........##...##.....##
.##....##.##.....##.##...###...##.##.........##...##.....##
..######...#######..##....##....###........######.########.
"""

class Conv1DTranspose(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, strides=1, padding='same', activation=None):
    super(Conv1DTranspose, self).__init__()
    self._layer = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(kernel_size,1),
                                                  strides=(strides,1), padding=padding, 
                                                  activation=activation)

  def call(self, x):
    x = self._layer(x)
    return x


def tf_nan_to_num(x, num=0.):
    return tf.where(tf.math.is_nan(x), tf.ones_like(x) * num, x)



"""
.##.....##..#######..########..########.##......
.###...###.##.....##.##.....##.##.......##......
.####.####.##.....##.##.....##.##.......##......
.##.###.##.##.....##.##.....##.######...##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##..#######..########..########.########
"""

class AEFIT1(tf.keras.Model):

    def __init__(self, feature_dim=100, latent_dim=20, latent_intervals=None):
        super(AEFIT1, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        if latent_intervals is None:
            x1 = []
            x2 = []
            for i in range(latent_dim):
                x1.append( i/latent_dim )
                x2.append( (i+1)/latent_dim )
            latent_intervals = np.array(list(zip(x1,x2)))
        self._voronoi = tf.convert_to_tensor(latent_intervals)
        self.set_model()        
        h = tf.signal.hamming_window(21)
        self._h = tf.reshape(h, [int(h.shape[0]), 1, 1])
        

    def set_model(self):
        feature_dim = self.feature_dim
        latent_dim = self.latent_dim
        ## INFERENCE ##
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(3*latent_dim,)),
            tf.keras.layers.Dense(3*latent_dim, activation=tf.nn.relu),
            tf.keras.layers.Dense(int(feature_dim/5), activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(int(feature_dim/5),1,1)),
            Conv1DTranspose( filters=64, kernel_size=30, strides=5, activation='relu'),
            Conv1DTranspose( filters=1, kernel_size=1, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(feature_dim,)),
            ]
        )

        ## GENERATION ##
        # self.generative_net = tf.keras.Sequential([
        #     tf.keras.layers.Input(shape=(feature_dim,)),
        #     tf.keras.layers.Reshape(target_shape=(feature_dim,1,)),
        #     tf.keras.layers.Conv1D(
        #         filters=64, kernel_size=20, activation='relu'),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(units=feature_dim, activation=tf.nn.relu),
        #     tf.keras.layers.Dense(units=3*latent_dim, activation=tf.nn.relu),
        #     tf.keras.layers.Dense(units=3*latent_dim, activation=tf.nn.relu),
        # ]
        # )
        self.inference_net.build()
        # self.generative_net.build()

    def compute_loss(self, batch_data):
        x,te,att = batch_data
        s = self.encode(x, te)    
        # X,TE = self.decode(s)

        # ACTIVE POINTS FIT
        # l0x = tf.losses.mean_squared_error(tf.boolean_mask(x,att), tf.boolean_mask(X,att))
        # l0y = tf.losses.mean_squared_error(tf.boolean_mask(te,att), tf.boolean_mask(TE,att))

        # # ALL POINTS FIT with s
        # X1 = tf.reshape(X,[-1])
        # T1 = tf.reshape(TE,[-1])
        # X1_id = tf.cast( (X1 + 0.4) / 0.8 * 100 , dtype = tf.int32 )
        # Y1 = tf.gather(tf.reshape(s,[-1]), X1_id)
        # l1 = tf.losses.mean_squared_error(T1, Y1)

        # ALL inputs with s
        x2 = tf.boolean_mask(x,att)
        X2 = tf.cast( (x2 + 0.4) / 0.8 * 100 , dtype = tf.int32 )
        Y2 = tf.gather(tf.reshape(s,[-1]), X2)
        l2 = tf.losses.mean_squared_error(tf.boolean_mask(te,att), Y2)        
        return  l2


    def set_dataset(self, ds):
        import copy        
        from sklearn.preprocessing  import normalize
        self._ds = copy.deepcopy(ds)
        self._k = self._ds.rebalance_prel()
        self._ds.set_null(np.nan)
        p_min = np.nanmin(self._ds['prel'], axis=0)
        p_max = np.nanmax(self._ds['prel'], axis=0)
        self._voronoi = np.array(list(zip(p_min,p_max)))
        # self._min = np.nanmin(self._ds['prel'])
        # self._max = np.nanmax(self._ds['prel'])
        # normalize te
        self._ds['te'] /= np.nanmax(self._ds['te'])
        return self._voronoi

    def dualize(self, X): 
        v = self._voronoi
        dx = v[:,1] - v[:,0]
        xL = (X-v[:,0]) / dx
        xH = (v[:,1]-X) / dx        
        dx_m = dx==0.
        dx_m = tf.reshape(tf.tile(dx_m,[len(X)]),[len(X),-1])        
        xL = tf.where(dx_m,tf.ones_like(xL) - 0.01,xL)
        xH = tf.where(dx_m,tf.zeros_like(xH) + 0.01,xH)
        xL = tf.where(tf.math.is_nan(X),tf.ones_like(xL) * np.nan,xL)
        xH = tf.where(tf.math.is_nan(X),tf.ones_like(xH) * np.nan,xH)
        return xL,xH

    def undualize(self, xL, xH):
        v = self._voronoi
        one = np.ones(self.latent_dim)
        dx = v[:,1] - v[:,0]
        x = (xL + one - xH) / 2.
        x = x * dx + v[:,0]
        return x

    def encode(self, x, te):
        xL,xH = self.dualize(x)
        X = tf.concat([xL,xH,te],-1)
        X = tf.convert_to_tensor(X)
        X = tf_nan_to_num(X)
        s = self.inference_net.call(X)
        
        # smooth
        s = tf.expand_dims(s, -1)
        s = tf.nn.conv1d(s, self._h, 1, 'SAME')
        s = tf.squeeze(s)
        return s

    def decode(self, s, act=None):
        xL, xH, te = tf.split(self.generative_net.call(s), num_or_size_splits=3, axis=1)
        x = self.undualize(xL,xH)
        if act is not None:
            x = tf.where(tf.math.logical_not(act), tf.ones_like(x) * np.nan, x)
            te = tf.where(tf.math.logical_not(act), tf.ones_like(te) * np.nan, te)
        te = te 
        return x, te

    def test_dual(self, X):
        x,_,act = X
        xL,xH = self.dualize(x)
        xp = self.undualize(xL,xH)
        xp = tf.where(tf.math.logical_not(act), tf.ones_like(xp) * np.nan, xp)
        return x,xp,xL,xH


    ## DATASET READER ##
    def get_tf_dataset(self):
        types = np.float, np.float, np.bool
        shape = ((20,),(20,),(20,))
        def gen():
            import itertools
            for i in itertools.count(0):
                if i < len(self._ds):
                    qsh = self._ds[i]
                    act = np.isfinite(qsh.prel)
                    yield qsh.prel, qsh.te, act
                else:
                    return
        return tf.data.Dataset.from_generator(gen, types, shape)



"""
..#######..########..########.####.##.....##.####.########.########.########.
.##.....##.##.....##....##.....##..###...###..##.......##..##.......##.....##
.##.....##.##.....##....##.....##..####.####..##......##...##.......##.....##
.##.....##.########.....##.....##..##.###.##..##.....##....######...########.
.##.....##.##...........##.....##..##.....##..##....##.....##.......##...##..
.##.....##.##...........##.....##..##.....##..##...##......##.......##....##.
..#######..##...........##....####.##.....##.####.########.########.##.....##
"""


optimizer = tf.keras.optimizers.Adam(1e-4)  

def compute_gradients(model, x, fn = None):    
    if fn is None:
        fn = model.compute_loss
    with tf.GradientTape() as tape:
        loss = fn(x)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))



"""
.########.########.....###....####.##....##.####.##....##..######..
....##....##.....##...##.##....##..###...##..##..###...##.##....##.
....##....##.....##..##...##...##..####..##..##..####..##.##.......
....##....########..##.....##..##..##.##.##..##..##.##.##.##...####
....##....##...##...#########..##..##..####..##..##..####.##....##.
....##....##....##..##.....##..##..##...###..##..##...###.##....##.
....##....##.....##.##.....##.####.##....##.####.##....##..######..
"""



def see_test(model, batch=100):
    import seaborn as sns
    it = model.get_tf_dataset().batch(batch).make_one_shot_iterator()
    test_data = it.get_next()
    plt.figure(1)
    plt.figure(2)
    plt.ion()
    x,te,att = test_data
    s = model.encode(x, te)
    # X,TE = model.decode(s)
    _int = np.linspace(-0.4,0.4,100)
    for i in range(len(x)):
        plt.figure(1)
        plt.clf()
        sns.scatterplot(x[i],te[i])
        plt.plot(_int,s[0])
        plt.pause(0.05)


def test_training(model, epochs=10, batch=10):
    import seaborn as sns
    from scipy import interpolate

    it = model.get_tf_dataset().batch(batch).make_one_shot_iterator()
    test_data = it.get_next()
    plt.figure(1)
    plt.figure(2)
    plt.ion()

    for epoch in range(epochs):
        count = 0
        it = model.get_tf_dataset().batch(batch).make_one_shot_iterator()
        for train_x in it:            
            gradients, loss = compute_gradients(model, train_x, model.compute_loss)
            apply_gradients(optimizer, gradients, model.trainable_variables)
            count += 1

            if count % 10 == 0:
                print('%d-%d loss: %f'%(epoch,count,tf.reduce_mean(loss)))
                x,te,att = test_data
                s = model.encode(x, te)
                # X,TE = model.decode(s,att)
                # X1,TE1 = model.decode(s)

                plt.figure(1)
                plt.clf()
                sns.scatterplot(x[0],te[0])
                # sns.scatterplot(X[0],TE[0])
                # sns.scatterplot(X1[0],TE1[0])
                
                
                plt.figure(2)                
                plt.clf()                                                
                _int = np.linspace(-0.4,0.4,100)
                plt.plot(_int,s[0])
                
                # x1 = tf.cast( (X1 + 0.4) / 0.8 * 100 , dtype = tf.int32 )
                # Y1 = tf.gather(s,x1[0],axis=1)
                sns.scatterplot(x[0],te[0])
                # sns.scatterplot(X1[0],Y1[0])
                # sns.scatterplot(X1[0],TE1[0])
                plt.pause(0.05)

        #     loss = tf.keras.metrics.Mean()
        #     for test_x in test_dataset:
        #     loss(compute_loss(model, test_x))
        #     elbo = -loss.result()
        #     display.clear_output(wait=False)
        #     print('Epoch: {}, Test set ELBO: {}, '
        #         'time elapse for current epoch {}'.format(epoch,
        #                                                     elbo,
        #                                                     end_time - start_time))
        #     generate_and_save_images(model, epoch, random_vector_for_generation)



