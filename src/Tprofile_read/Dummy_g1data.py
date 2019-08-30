from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import abc

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors 

import copy
import models.base

import Hunch_utils  as Htls



class ComposableAccess(Htls.Struct):
    __metaclass__ = abc.ABCMeta            
    _data = np.empty(1,dtype=None)

    def __init__(self, *ref):
        super().__init__(self, _data = ref[0])
        
    def __getattr__(self, name):
        return self._data[name]

    def __setattr__(self, name, val):
        self._data[name] = val
    
    def __getitem__(self, key):        
        return self.get_item_byname(key)

    def get_item_byname(self, key):
        fields = key.split('~')
        if len(fields) > 1: 
            val = np.concatenate([ np.atleast_1d(self.get_item_byname(k)) for k in fields ])
        else:
            try:    val = self._data[key]
            except: val = np.nan #np.full([len(self)], self._null)                
        return val   


class Dummy_g1data(models.base.Dataset):

    kinds = [
        {'mean': [0.2,0.8], 'sigma': [0.1,0.1], 'gain': [1,1] },        
        {'mean': [0.8], 'sigma': [0.1], 'gain': [0.5] },
        {'mean': [0.2], 'sigma': [0.1], 'gain': [0.5] },
        {'mean': [0.5], 'sigma': [0.2], 'gain': [1] },
        {'mean': [0.5], 'sigma': [0.2], 'gain': [0.5] },
    ]


    def __init__(self, counts=60000, size=20, noise_var=0., nanprob=None, nanmask=None, fixed_nanmask=None):
        self._counts = counts
        self._size = size
        self._noise = noise_var
        self._nanmask = None
        self._nanprob = None
        self._fixed_nanmask = None
        if nanmask is not None:
            self._nanmask = np.array(nanmask)
        if nanprob is not None:
            self._nanprob = np.array(nanprob)
        if fixed_nanmask is not None:
            self._fixed_nanmask = np.array(fixed_nanmask)
        self._null = np.nan
        self._dataset = None




    def buffer(self, counts = None):
        if counts is None: counts = self._counts
        else             : self._counts = counts
        size = self._size
        dtype = np.dtype ( [  ('x', '>f4', (size,) ),
                              ('y', '>f4', (size,) ),
                              ('y_min', np.float32),
                              ('y_max', np.float32),
                              ('y_mean', np.float32),
                              ('y_median', np.float32),
                              ('l_magic', np.float32),
                              ('l_mean_gain', np.float32),
                              ('l_mean_sigma', np.float32),
                              ('l', np.int32),
                           ] )
        ds = np.empty([counts], dtype=dtype)
        for i in range(counts):
            s_pt,l = self.gen_pt(i)
            ds[i] = (
                     s_pt[:,0], s_pt[:,1], 
                     np.nanmin(s_pt[:,1]),
                     np.nanmax(s_pt[:,1]),
                     np.nanmean(s_pt[:,1]),
                     np.nanmedian(s_pt[:,1]),
                     float(l)/len(self.kinds),
                     np.mean(self.kinds[l]['gain']),
                     np.mean(self.kinds[l]['sigma']),
                     l, 
                    )
        self._dataset = ds
        return self
    
    def clear(self):
        self._dataset = None

    def __len__(self):
        return self._counts

    # return by reference
    def __getitem__(self, key):
        assert self._dataset is not None, 'please fill buffer first'
        if isinstance(key, int):
            return ComposableAccess(self._dataset[key])
            # return self._dataset[key]
        elif isinstance(key, range):
            return self._dataset[key]
        elif isinstance(key, slice):
            ds = copy.deepcopy(self)
            ds._dataset = self._dataset[key]
            ds._counts = len(ds._dataset)
            return ds
        elif isinstance(key, str):
            try:    val = self._dataset[:][key]
            except: val = np.full([self._counts], self._null)
            return val
        elif isinstance(key, tuple):
            val = [ self[:][k] for k in key ]
            return val
        else:
            print("not supported index: ",type(key))



    # set by reference
    def __setitem__(self, key, value):
        assert self._dataset is not None, 'please fill buffer first'
        if isinstance(key, int):
            self._dataset[key] = value
        elif isinstance(key, range) or isinstance(key, slice):
            self._dataset[key] = value
        elif isinstance(key, str):
            try: self._dataset[:][key] = value
            except: print('WARNING: field not found')
        else:
            print("not supported index: ",type(key))


    @property
    def dim(self):
        return self._size

    @property
    def size(self):
        return self._counts

    def gen_pt(self, id=None, x=None, kind=None):
        def gauss(x, m, s, g):
            return np.abs(np.exp(-np.power(x-m, 2.) / (2 * np.power(s, 2.))) * g + np.random.normal(0,self._noise,1))
        if self._dataset is not None and id is not None:
            data = self._dataset[id]
            return np.stack([data['x'],data['y']], axis=1), data['l']
        else:
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
            
            mask = np.zeros_like(x)
            if self._nanprob is not None:
                mask = np.random.uniform(size=self._size)
                mask = (mask < self._nanprob).astype(float)

            # if self._nanmask is not None:
            #     mask = self._nanmask & np.random.randint(2, size=self._size)
            #     if self._fixed_nanmask is not None:
            #         mask = mask | self._fixed_nanmask
            
            x[mask > 0] = np.nan
            y[mask > 0] = np.nan
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
        

    def tf_tuple_compose(self, fields=[]):
        def gen():
            import itertools
            for i in itertools.count(0):
                if i < len(self):
                    pt = self[i]
                    yield tuple([ pt[n] for n in fields])
                else:
                    return
        d0 = tuple([ self[0][n] for n in fields])
        types = tuple([tf.convert_to_tensor(x).dtype for x in d0])
        shape = tuple([np.shape(x) for x in d0])
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