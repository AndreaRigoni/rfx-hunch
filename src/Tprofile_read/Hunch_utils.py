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









# """
# ...##.##......##.....##....###....####.##....##
# ...##.##......###...###...##.##....##..###...##
# .#########....####.####..##...##...##..####..##
# ...##.##......##.###.##.##.....##..##..##.##.##
# .#########....##.....##.#########..##..##..####
# ...##.##......##.....##.##.....##..##..##...###
# ...##.##......##.....##.##.....##.####.##....##
# """

# def tsne_analysis():
#     print("tf  version: %s" % tf.__version__)
#     # print("mds version: %s" % mds.__version__)
#     qsh = QSH_Dataset()
#     qsh.load('te_db_1.npy')
#     qsh.shuffle()

#     tsne = tSNE()
#     tsne.random = 42
    
#     clst = Clustering()
#     clst.n_clusters = 5

#     Y = tsne.draw((qsh['te'][0:1000],qsh['tcentro'][0:1000]))
#     L = clst(Y)
#     clst.draw()

#     fig = plt.figure()
#     fig.clf() 
    
#     cm = colors.ListedColormap(['k','b','y','g','r']) 
#     for i in range(1000):
#         c = np.linspace(0,255,)
#         te = qsh['te'][i]
#         plt.plot(te,'-', color=cm(L[i]), linewidth=0.2) 

#     plt.ion()
#     plt.show()



# if __name__ == '__main__':
#     tsne_analysis()
