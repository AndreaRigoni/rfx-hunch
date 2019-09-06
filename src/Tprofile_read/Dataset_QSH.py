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

from collections import OrderedDict

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors 

import copy
import models.base

import Hunch_utils  as Htls




class QSH(Htls.Struct):
    __metaclass__ = abc.ABCMeta            
    _dtype = np.dtype ( [  ('label','S10'),
                           ('i_qsh', np.int32 ),
                           ('n_ok', np.int32 ),
                           ('tbordo','>f4' ),
                           ('tcentro','>f4' ),
                           ('pos','>f4' ),
                           ('grad','>f4' ),
                           ('prel','>f4', (20,) ),
                           ('rho','>f4', (20,) ),
                           ('te','>f4', (20,) ),
                        ] )
    _data = np.empty(1,dtype=_dtype)

    def __init__(self, *ref):
        super().__init__(self, _data = ref[0])
        
    def __getattr__(self, name):
        return self._data[name]

    def __setattr__(self, name, val):
        self._data[name] = val
    
    def __getitem__(self, key):        
        return self._get_item_byname(key)

    def _get_item_byname(self, key, dim=None):
        key = key.split(':')
        if len(key) > 1: 
            dim = int(key[1])
        key = key[0]
        fields = key.split('~')
        if len(fields) > 1: 
            val = np.concatenate([ np.atleast_1d(self._get_item_byname(k, dim)) for k in fields ], axis=-1)
        else:
            try: val = self._data[key]
            except: 
                try: val = getattr(self, key)
                except: val = np.nan 
            if dim is not None: val = val[0:dim]            
        return val
        
    @property
    def pulse(self):
        # almeno verifico che sia un impulso di RFX-mod	
        shot = int(self.label.split(b'_')[0])
        if ( shot < 15600 or shot > 39391 ) :
	        raise UserWarning('Not a RFX-mod shot')
        return shot

    @property
    def start(self):
        ''' get start of the profile in relative time [ms]*1E-1
        '''
        return int(self.label.split(b'_')[1])

    @property
    def Bt_rm(self):
        abs = self['absBt_rm']
        arg = self['argBt_rm']
        re,im = abs * (np.cos(arg), np.sin(arg))
        return np.concatenate([re,im], axis=-1 )

    @property
    def Br_rm(self):
        abs = self['absBr_rm']
        arg = self['argBr_rm']
        re,im = abs * (np.cos(arg), np.sin(arg))
        return np.concatenate([re,im], axis=-1 )

    @property
    def Br_rs(self):
        abs = self['absBr_rs']
        arg = self['argBr_rs']
        re,im = abs * (np.cos(arg), np.sin(arg))
        return np.concatenate([re,im], axis=-1 )


    def plot_countour(self, ax = None):
        # contour della topologia della mappa di flusso
        if ax is None:
            fig = plt.figure( 'Flux' )
            fig.set_size_inches( 6, 5 )
            fig.clf()
            fig.set_dpi(150)
            ax = plt.gca()
        fig.subplots_adjust( top=0.95, bottom=0.08, left=0.08, right=0.95, hspace=0.2, wspace=0.2 )
        cntr1 = ax.contour( self.xxg, self.yyg, self.mapro, levels=24 )        
        # fig.colorbar(cntr1, ax=ax)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title( r'%5d_%04d' % ( self.pulse, self.start ) )








class Dataset_QSH(models.base.Dataset):
                
    def __init__(self, dim=20):
        self._dataset = None
        self._range   = None
        self._dim     = dim
        self._balance = 0
        self._null    = -1
        self.batch    = None

    # return by reference
    def __getitem__(self, key):
        if isinstance(key, int):
            return QSH(self._dataset[key])
        elif isinstance(key, range):
            return self._dataset[key]
        elif isinstance(key, slice):
            return self._dataset[key]
        elif isinstance(key, str):
            return QSH(self._dataset[:])[key]
        elif isinstance(key, tuple):
            return [ self[k] for k in key ]
        else:
            print("not supported index: ",type(key))



    # set by reference
    def __setitem__(self, key, value):
        if isinstance(key, int):
            self._dataset[key] = value
        elif isinstance(key, range) or isinstance(key, slice):
            self._dataset[key] = value
        elif isinstance(key, str):
            self._dataset[:][key] = value
        else:
            print("not supported index: ",type(key))



    # return a copy
    @property
    def data(self):
        if range is not None:
            return np.rec.array(self._dataset)
        else:
            return np.rec.array(self._dataset[self._range])

    @property
    def dictionary(self):
        a = self.data
        return {name:a[name] for name in a.dtype.names}

    @property
    def ls(self):
        return self.data.dtype.names

    def __len__(self):
        return len(self._dataset)
            
    def get_dim(self):
        return self._dim

    def set_dim(self, dim):
        self._dim = dim
        # if self.is_balanced and self._balance != dim:
        #     self.rebalance_prel(dim)

    def is_balanced(self):
        return self._balance != 0

    dim = property(get_dim,set_dim)

    def load(self, file):
        try:
            self._dataset = np.load( file )            
        except:
            print("error loading np database")

    def save(self, file):
        try:            
            np.save(file, self._dataset)
        except:
            print("error saving np database")

    def clean_array(self, a):
        if np.isnan(self._null) or np.isinf(self._null):
            return a[np.isfinite(a)]
        else:
            return a[a!=self._null]

    def clean_up_poorcurves(self, count=1):
        ds = [el for el in self._dataset if len( self.clean_array(el['prel']) ) > count]
        self._dataset = np.array(ds, dtype=self._dataset.dtype)

    def filter_number_set(self, count):
        ds = [el for el in self._dataset if len( self.clean_array(el['prel']) ) == count]
        self._dataset = np.array(ds, dtype=self._dataset.dtype)
        self.dim = count

    def unbias_mean(self, mean=None, axis='te'):
        assert np.isnan(self.get_null())
        if mean is None:
            mean = np.nanmean(self[axis])
        for x in self._dataset:
            y = x[axis]
            x[axis] = y-np.nanmean(y)+mean
        
    def clip_values(self, a_min, a_max, axis='te'):
        self._dataset['te'] = np.clip(self._dataset['te'], a_min=a_min, a_max=a_max )

    def get_null(self):
        return self._null

    def set_null(self, s_out=np.nan, datasets=None):
        if datasets is None:
            datasets = ['prel', 'rho', 'te']
        for ds in datasets:
            el = self[ds]
            el[self.is_null(el)] = s_out
        self._null = s_out    

    def is_null(self, x):
        if np.isnan(self._null):
            return np.isnan(x)
        else:
            return x==self._null

    null = property(get_null,set_null)

    def shuffle(self):
        np.random.shuffle(self._dataset)

    def rebalance_prel(self, n_clusters=20):
        from sklearn.cluster import KMeans            
        prel = self.clean_array(self.data['prel']).reshape(-1,1)
        k = KMeans(n_clusters=n_clusters, random_state=0)
        k.fit(prel)        
        idx = np.argsort(k.cluster_centers_.sum(axis=1))
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(k.n_clusters)
        for el in self:
            pr_c = self.clean_array(el.prel)
            te_c = self.clean_array(el.te)
            id = lut[k.predict(pr_c.reshape(-1,1))]
            el.prel[:] = self.null
            el.prel[id] = pr_c
            el.te[:] = self.null
            el.te[id] = te_c
        self._is_balanced = True
        self._dim = n_clusters
        return k

    def rebalance_rho(self ,n_clusters=20):
        from sklearn.cluster import KMeans            
        rho = self.clean_array(self.data['rho']).reshape(-1,1)
        k = KMeans(n_clusters=n_clusters, random_state=0)
        k.fit(rho)
        idx = np.argsort(k.cluster_centers_.sum(axis=1))
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(k.n_clusters)
        for el in self:
            pr_c = self.clean_array(el.rho)
            te_c = self.clean_array(el.te)
            id = lut[k.predict(pr_c.reshape(-1,1))]
            el.rho[:] = self.null
            el.rho[id] = pr_c
            el.te[:] = self.null
            el.te[id] = te_c
        self._is_balanced = True
        self._dim = n_clusters
        return k

    def set_normal(self, fields=['prel','te','rho']):
        # assert self._null == np.nan
        _null = self._null 
        if not np.isnan(self._null):
            print("Warning normilizing not a null=nan qsh... this will not normalize null value")
            self.set_null(np.nan)
        def normalize(axis):
            data = self[axis]
            _min = np.nanmin(data)
            _max = np.nanmax(data)
            data = data - np.nanmean(data)
            self[axis] = data / np.max( np.abs(_min), np.abs(_max) )
        for n in fields:
            normalize(n)
        self.set_null(_null)


    def set_normal_positive(self, fields=['prel','te','rho']):
        # assert self._null == np.nan
        _null = self._null 
        if not np.isnan(self._null):
            print("Warning normilizing not a null=nan qsh... this will not normalize null value")
            self.set_null(np.nan)
        def normalize(axis):
            data = self[axis]
            _min = np.nanmin(data)
            _max = np.nanmax(data)
            self[axis] = (data-_min)/(_max-_min)
        for n in fields:
            normalize(n)
        self.set_null(_null)




    ## WORKING ON ....
    # def set_stats(self, fields=['prel','te','rho']):
    #     import scipy.stats
    #     stats = {}
    #     for n in fields:
    #         _null = self._null 
    #         if not np.isnan(self._null):
    #             print("Warning normilizing not a null=nan qsh... this will not normalize null value")
    #             self.set_null(np.nan)
    #         stats[n] = scipy.stats.describe(self[n], nan_policy='omit')
    #         self.set_null(_null)





    def missing_values_mask(self, datasets=None):
        from sklearn.impute import MissingIndicator
        indicator = MissingIndicator(missing_values=self.null)
        if datasets is None:
            datasets = ['prel', 'rho', 'te']
        mv_mask = {
            'prel': [],
            'rho': [],
            'te': [],
        }
        for ds in datasets:
            el = self[ds]
            mv_mask[ds] = indicator.fit_transform(el)
        return mv_mask

    def get_tf_dataset(self):
        types = np.float, np.float, np.bool
        shape = ((20,),(20,),(20,),)
        def gen():
            import itertools
            for i in itertools.count(0):
                if i < len(self):
                    qsh = self[i]
                    act = np.isfinite(qsh.prel)
                    yield qsh.prel, qsh.te, act
                else:
                    return
        return tf.data.Dataset.from_generator(gen, types, shape)

    def get_tf_dataset_array(self):
        types = tf.float32, tf.int32
        shape = (2*self.dim,),(self.dim)
        def gen():
            import itertools
            for i in itertools.count(0):
                if i < len(self):
                    qsh = self[i]
                    act = np.isfinite(qsh.prel)
                    yield np.concatenate([qsh.prel[0:self.dim], qsh.te[0:self.dim]]), act[0:self.dim]
                else:
                    return
        return tf.data.Dataset.from_generator(gen, types, shape)
        

    ## REWRITE WITHOUT TF
    def plot_hisotgrams(self):
        import seaborn as sns
        x = range(20)
        def count_not_nan(xi):
            y = tf.boolean_mask(xi, tf.math.is_finite(xi))
            return len(y)
        y = [ self['te'][:,i] for i in range(20) ]
        Y = tf.map_fn(lambda xi: count_not_nan(xi), tf.convert_to_tensor(y), dtype=tf.int32 )
        plt.figure('not nan histogram')
        plt.clf()
        plt.bar(x,Y)
        
        yh = [ len(y[np.isfinite(y)]) for y in self['te'] ]
        plt.figure('not nan len distribution')
        plt.clf()
        sns.distplot(yh)
        yh_max = np.max(yh)
        print("this should be shriked to: ",yh_max)

    def tf_tuple_compose(self, fields=[]):
        def clip(x):
            try: 
                if len(x) > self.dim: x=x[0:self.dim]
            except: pass
            return x
        def gen():
            import itertools
            for i in itertools.count(0):
                if i < len(self):
                    qsh = self[i]
                    yield tuple([ clip(qsh[n]) for n in fields])
                else:
                    return
        # d0 = [x for x in gen()][0]
        d0 = tuple([ clip(self[0][n]) for n in fields])
        types = tuple([tf.convert_to_tensor(x).dtype for x in d0])
        shape = tuple([np.shape(x) for x in d0])
        return tf.data.Dataset.from_generator(gen, types, shape)




    # PROPERTIES
    ds_tuple = property(get_tf_dataset)
    ds_array = property(get_tf_dataset_array)


