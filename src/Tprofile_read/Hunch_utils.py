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
import models






#########    .##.....##.########.####.##........######.     #########
#########    .##.....##....##.....##..##.......##....##     #########
#########    .##.....##....##.....##..##.......##......     #########
#########    .##.....##....##.....##..##........######.     #########
#########    .##.....##....##.....##..##.............##     #########
#########    .##.....##....##.....##..##.......##....##     #########
#########    ..#######.....##....####.########..######.     #########


class Struct:
    def __init__ (self, *argv, **argd):
        if len(argd):
            # Update by dictionary
            self.__dict__.update (argd)
        else:
            # Update by position
            attrs = filter (lambda x: x[0:2] != "__", dir(self))
            for n in range(len(argv)):
                setattr(self, attrs[n], argv[n])



class Stat2(Struct):
    m_min    = 0.
    m_max    = 0.
    m_mean   = 0.
    m_M2     = 0.
    m_count  = 0 
    def __add__(self, data):
        self.m_count += 1
        if data < self.m_min: self.m_min = data
        if data > self.m_max: self.m_max = data
        delta = float(data) - self.m_mean
        self.m_mean += delta/self.m_count # approximation here
        self.m_M2   += delta*(data - self.m_mean)
        return self
    
    def __len__(self):
        return self.m_count

    def variance(self):
        if self.m_count < 2:
            return 0.
        else:
            return self.m_M2/(self.m_count -1)

    def rms(self):
        return np.sqrt(self.variance())
    
    def mean(self):
        return self.m_mean
    
    def min(self):
        return self.m_min

    def max(self):
        return self.m_max

    @staticmethod
    def test():
        s2 = Stat2()
        for i in range(10000):
            d = np.random.normal(10.,5.)
            s2 += d
        print('size = ', len(s2))
        print('mean = ', s2.mean())
        print('rms  = ', s2.rms())




# ///////////////////////////////////////////////////////////////////////////////////////////// #
# //  PLOT UTILS   //////////////////////////////////////////////////////////////////////////// #
# ///////////////////////////////////////////////////////////////////////////////////////////// #

class utils:

    # colab friendly graphs
    @staticmethod
    def plt_init():
        SMALL_SIZE = 2
        MEDIUM_SIZE = 4
        BIGGER_SIZE = 6
        #
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        #
        plt.rcParams['figure.figsize'] = [10, 10]
        plt.rcParams['figure.dpi'] = 200

    # Print iterations progress
    @staticmethod
    def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        # print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        # Print New Line on Complete
        if iteration == total: 
            print()


    @staticmethod
    def plot(y, *args, fig=None):
        if not fig:
            fig = plt.figure()
            fig.set_size_inches( 8, 5.25 )
        plt.plot(y,args)
        plt.draw()        
        return fig



"""
.########..........######..##....##.########
....##............##....##.###...##.##......
....##............##.......####..##.##......
....##....#######..######..##.##.##.######..
....##..................##.##..####.##......
....##............##....##.##...###.##......
....##.............######..##....##.########
"""

class tSNE(Struct):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    tSNE = TSNE(n_components=2)
    data_X = None
    data_Y = None
    data_C = None
    data_S = None
    fig  = None
    interactive = False
    n_components  = property()
    perplexity    = property()
    learning_rate = property()
    n_iter        = property()

    def __setattr__(self, name, value):
        if name in self.tSNE.__dict__:
            setattr(self.tSNE, name, value)
            if self.interactive == True:
                self.update()
                self.draw()
        else:
            object.__setattr__(self, name, value)
                    
    def __getattr__(self, name):
        if name in self.tSNE.__dict__:
            return getattr(self.tSNE, name)        

    def __call__(self, data):
        if isinstance(data, tuple):
            if len(data) >= 1:
                self.data_X = data[0]
            if len(data) >= 2:
                self.data_C = data[1]
            if len(data) >= 3:
                self.data_S = data[2]
        else:
            self.data_X = data
            self.data_C = 'grey'
            self.data_S = 10
        return self.update()
    
    def update(self):            
        if self.data_X is not None:
            self.data_Y = self.tSNE.fit_transform(self.data_X)
        if isinstance(self.data_C, Clustering):
            self.data_C = self.data_C(self.data_Y)
        if isinstance(self.data_S, Clustering):
            self.data_S = self.data_S(self.data_C)
        return self.data_Y

    def draw(self, data=None):
        if data is not None:
            self.__call__(data)
            self.update()
        self.draw_plt()
        return self.data_Y
    
    def draw_plt(self):            
        if self.fig is None:
            self.fig = plt.figure()
            self.fig.set_size_inches( 8, 5.25 )            
        plt.figure(self.fig.number)
        plt.clf()
        shape = np.array(self.data_Y).shape
        if   shape[1] == 1:
            plt.plot(self.data_Y, '-b')
        elif shape[1] == 2:
            plt.scatter(self.data_Y[:,0],self.data_Y[:,1], 
                        c=self.data_C, s=self.data_S, marker=',')
        elif shape[1] == 3:
            pass
        plt.draw()
    

"""
..######..##.......##.....##..######..########.########.########.
.##....##.##.......##.....##.##....##....##....##.......##.....##
.##.......##.......##.....##.##..........##....##.......##.....##
.##.......##.......##.....##..######.....##....######...########.
.##.......##.......##.....##.......##....##....##.......##...##..
.##....##.##.......##.....##.##....##....##....##.......##....##.
..######..########..#######...######.....##....########.##.....##
"""

class Clustering(Struct):
    from sklearn import cluster
    clust = cluster.AgglomerativeClustering()
    
    interactive = True
    data  = None
    fig   = None

    n_clusters = property()
    labels_    = property()

    def __setattr__(self, name, value):
        if name in self.clust.__dict__:
            setattr(self.clust, name, value)
            if self.interactive == True:
                self.update()
        else:
            object.__setattr__(self, name, value)
                    
    def __getattr__(self, name):
        if name in self.clust.__dict__:
            return getattr(self.clust.__dict__, name)

    def __call__(self, data):
        self.data = np.array(data)
        return self.update()

    def update(self):
        self.clust.fit(self.data)
        if self.fig is not None:
            self.draw()
        return self.clust.labels_

    def draw(self, data=None):
        if data is not None:
            self.__call__(data)
            self.update()
        self.draw_plt()
        return self.clust.labels_

    def draw_plt(self):            
        if self.fig is None:
            self.fig = plt.figure()
            self.fig.set_size_inches( 8, 5.25 )            
        plt.figure(self.fig.number)
        plt.clf()
        shape = np.array(self.data).shape
        if   shape[1] == 1:
            plt.plot(self.data, '-b')
        elif shape[1] == 2:
            plt.scatter(self.data[:,0],self.data[:,1], 
                        c=self.clust.labels_)
        elif shape[1] == 3:
            pass
        plt.draw()




"""
..#######...######..##.....##
.##.....##.##....##.##.....##
.##.....##.##.......##.....##
.##.....##..######..#########
.##..##.##.......##.##.....##
.##....##..##....##.##.....##
..#####.##..######..##.....##
"""


class QSH(Struct):
    __metaclass__ = abc.ABCMeta            
    _dtype = np.dtype ( [  ('label','S10'),
                                ('i_qsh', np.int32 ),
                                ('tbordo','>f4' ),
                                ('tcentro','>f4' ),
                                ('pos','>f4' ),
                                ('grad','>f4' ),
                                ('n_ok', np.int32 ),
                                ('prel','>f4', (20,) ),
                                ('rho','>f4', (20,) ),
                                ('te','>f4', (20,) ),

							# SHEq map
							('mapro','>f4', (51,51) ),
							('xxg','>f4', (51,) ),
							('yyg','>f4', (51,) ),

							# Spectrum
							('n', '>i4', (10,) ),
							('absBt_rm', '>f4', (10,)),
							('argBt_rm', '>f4', (10,)),
							('absBr_rm', '>f4', (10,)),
							('argBr_rm', '>f4', (10,)),
							('absBt_rs', '>f4', (10,)),
							('argBt_rs', '>f4', (10,)),
							('absBr_rs', '>f4', (10,)),
							('argBr_rs', '>f4', (10,)),
							('absBr_rp', '>f4', (10,)),
							('argBr_rp', '>f4', (10,)),
							('absBr_max', '>f4', (10,)),
							('absFlux_max', '>f4', (10,)),

                            ] )
    _data = np.empty(1,dtype=_dtype)

    def __init__(self, *ref):
        super().__init__(self, _data = ref[0])

    def __getattr__(self, name):
        return self._data[name]

    def __setattr__(self, name, val):
        self._data[name] = val
    
    def get_pulse(self):
        # almeno verifico che sia un impulso di RFX-mod	
        shot = int(self.label.split(b'_')[0])
        if ( shot < 15600 or shot > 39391 ) :
	        raise UserWarning('Not a RFX-mod shot')
        return shot

    def get_start(self):
        ''' get start of the profile in relative time [ms]*1E-1
        '''
        return int(self.label.split(b'_')[1])
    

    label    = property()
    i_qsh    = property()
    tborder  = property()
    tcenter  = property()
    pos      = property()
    grad     = property()
    n_ok     = property()
    prel     = property()
    rho      = property()
    te       = property()
    pulse    = property(get_pulse)
    start    = property(get_start)
    
    mapro = property()
    xxg = property()
    yyg = property()
    
    n = property()
    absBt_rm = property()
    argBt_rm = property()
    absBr_rm = property()
    argBr_rm = property()
    absBt_rs = property()
    argBt_rs = property()
    absBr_rs = property()
    argBr_rs = property()
    absBr_rp = property()
    argBr_rp = property()
    absBr_max = property()
    absFlux_max = property()


    def plot_countour(self, ax = None):
        # contour della topologia della mappa di flusso
        if ax is None:
            fig = plt.figure( 'Flux' )
            fig.set_size_inches( 6, 5 )
            fig.clf()
            ax = plt.gca()
        fig.subplots_adjust( top=0.95, bottom=0.08, left=0.08, right=0.95, hspace=0.2, wspace=0.2 )
        ax.contour( self.xxg, self.yyg, self.mapro )
        ax.set_aspect('equal', adjustable='box')
        ax.set_title( r'%5d_%04d' % ( self.pulse, self.start ) )








class QSH_Dataset(models.base.Dataset):
                
    def __init__(self, dim=20):
        self._dataset = None
        self._range   = None
        self._dim     = dim
        self._balance = 0
        self._null    = -1

    # return by reference
    def __getitem__(self, key):
        if isinstance(key, int):
            return QSH(self._dataset[key])
        elif isinstance(key, range):
            return self._dataset[key]
        elif isinstance(key, slice):
            qsh = copy.deepcopy(self)
            qsh._dataset = self._dataset[key]
            return qsh
        elif isinstance(key, str):
            return self._dataset[:][key]
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
            x[axis] = y-np.nanmean(y)
        
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

    def set_normal_positive(self):
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
        normalize('prel')
        normalize('te')
        normalize('rho')
        self.set_null(_null)
        

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




    # PROPERTIES
    ds_tuple = property(get_tf_dataset)
    ds_array = property(get_tf_dataset_array)











"""
...##.##......##.....##....###....####.##....##
...##.##......###...###...##.##....##..###...##
.#########....####.####..##...##...##..####..##
...##.##......##.###.##.##.....##..##..##.##.##
.#########....##.....##.#########..##..##..####
...##.##......##.....##.##.....##..##..##...###
...##.##......##.....##.##.....##.####.##....##
"""

def tsne_analysis():
    print("tf  version: %s" % tf.__version__)
    # print("mds version: %s" % mds.__version__)
    qsh = QSH_Dataset()
    qsh.load('te_db_1.npy')
    qsh.shuffle()

    tsne = tSNE()
    tsne.random = 42
    
    clst = Clustering()
    clst.n_clusters = 5

    Y = tsne.draw((qsh['te'][0:1000],qsh['tcentro'][0:1000]))
    L = clst(Y)
    clst.draw()

    fig = plt.figure()
    fig.clf() 
    
    cm = colors.ListedColormap(['k','b','y','g','r']) 
    for i in range(1000):
        c = np.linspace(0,255,)
        te = qsh['te'][i]
        plt.plot(te,'-', color=cm(L[i]), linewidth=0.2) 

    plt.ion()
    plt.show()



if __name__ == '__main__':
    tsne_analysis()
