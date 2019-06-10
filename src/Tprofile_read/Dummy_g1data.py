from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import abc

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors 



class Dummy_g1data():

    kinds = [
        {'mean': [0.2,0.8], 'sigma': [0.1,0.1], 'gain': [1,1] },        
        {'mean': [0.8], 'sigma': [0.1], 'gain': [0.5] },
        {'mean': [0.2], 'sigma': [0.1], 'gain': [0.5] },
        {'mean': [0.5], 'sigma': [0.2], 'gain': [1] },
        {'mean': [0.5], 'sigma': [0.2], 'gain': [0.5] },
    ]

    def __init__(self, counts=20, size=20, noise_var=0.):
        self._counts = counts
        self._size = size
        self._noise = noise_var
    
    def __len__(self):
        return self._counts

    def gen_pt(self, id, x=None, kind=None):
        def gauss(x, m, s, g):
            return np.abs(np.exp(-np.power(x-m, 2.) / (2 * np.power(s, 2.))) * g + np.random.normal(0,self._noise,1))
        if x is None:
            x = np.sort(np.random.rand(self._size))
        y = np.zeros_like(x)        
        if kind is None:
            kind = np.random.randint(len(self.kinds))
        k = self.kinds[kind]
        if len(np.shape(k['mean'])) > 0:
            for m,s,g in np.stack([k['mean'],k['sigma'],k['gain']],axis=1):
                y += gauss(x,m,s,g)
        else:
            y = gauss(x,k['mean'],k['sigma'],k['gain'])
        return np.stack([x,y], axis=1), kind
    
    def get_tf_dataset_tuple(self):
        types = tf.float32, tf.float32, tf.int32
        shape = (self._size,),(self._size,),()
        def gen():
            import itertools
            for i in itertools.count(0):
                if i < len(self):
                    s_pt,l = self.gen_pt(i)
                    yield s_pt[:,0], s_pt[:,1], l
                else:
                    return
        return tf.data.Dataset.from_generator(gen, types, shape)

    def get_tf_dataset_array(self):
        types = tf.float32, tf.int32
        shape = (2*self._size,),()
        def gen():
            import itertools
            for i in itertools.count(0):
                if i < len(self):
                    s_pt, l = self.gen_pt(i)
                    yield np.concatenate([s_pt[:,0], s_pt[:,1]]), l
                else:
                    return
        return tf.data.Dataset.from_generator(gen, types, shape)
        
    ds_tuple = property(get_tf_dataset_tuple)
    ds_array = property(get_tf_dataset_array)



def test_gendata(g1=None):
    if g1 is None:
        g1 = Dummy_g1data()
    x,y,_ = g1.ds_tuple.batch(200).make_one_shot_iterator().get_next()
    plt.figure('g1')
    plt.clf()
    for x,y in np.stack([x,y],axis=1):
        plt.plot(x,y,'.')