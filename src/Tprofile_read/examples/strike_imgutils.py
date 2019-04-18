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

# FOR BENCHMARCH 
import time





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



def Stat2_test():
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

class Strike_PlotUtils:

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

    @staticmethod
    def show_images(img, xy='nan', xys='nan', xys2='nan', xys3='nan'):
        dim = len(img)
        sqd = int(np.sqrt(dim))
        sqx = sqd + (dim - sqd*sqd > 0 )
        sqy = sqd + (dim - sqd*sqd > sqd )
        fig, ax = plt.subplots(sqx,sqy)
        for i in range(dim):
            axi = ax[i%sqx, int(i/sqx)]
            axi.grid(False)
            axi.imshow(img[i], cmap="Greys")
            if xy != 'nan':                
                xy_i = xy[i]
                for j in range(0,len(xy_i),2):
                    x,y = xy_i[j:j+2]
                    rect   = patches.Rectangle((y-1,x-1),2,2, linewidth=1,edgecolor='r',fill=False)
                    axi.add_patch(rect)
            if xys != 'nan':                
                xys_i = xys[i]
                for j in range(0,len(xys_i),3):
                    x,y,s = xys_i[j:j+3]
                    rect   = patches.Rectangle((y-1,x-1),2,2, linewidth=1,edgecolor='r',fill=False)
                    circle = patches.Circle((y,x),s/10, fill=False, edgecolor="b")
                    axi.add_patch(rect)
                    axi.add_patch(circle)
            if xys2 != 'nan':                
                xys2_i = xys2[i]                
                for j in range(0,len(xys2_i),4):
                    x,y,s1,s2 = xys2_i[j:j+4]
                    rect   = patches.Rectangle((y-1,x-1),2,2, linewidth=1,edgecolor='r',fill=False)
                    axi.add_patch(rect)
                    ellipse = patches.Ellipse((y,x),s2,s1,0, fill=False, edgecolor="b")
                    axi.add_patch(ellipse)
            if xys3 != 'nan':                
                xys3_i = xys3[i]                
                for j in range(0,len(xys3_i),5):
                    x,y,s1,s2,s3 = xys3_i[j:j+5]
                    rect   = patches.Rectangle((y-1,x-1),2,2, linewidth=1,edgecolor='r',fill=False)
                    axi.add_patch(rect)
                    ellipse = patches.Ellipse((y,x),s2/10,s1/10,s3*180/np.pi, fill=False, edgecolor="b")
                    axi.add_patch(ellipse)
        plt.draw()
        return fig

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
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
        # Print New Line on Complete
        if iteration == total: 
            print()


# ///////////////////////////////////////////////////////////////////////////////////////////// #
# //  IMAGE UTILS   /////////////////////////////////////////////////////////////////////////// #
# ///////////////////////////////////////////////////////////////////////////////////////////// #


## ..######..########.########..####.##....##.########.........####.##.....##..######...##.....##.########.####.##........######.
## .##....##....##....##.....##..##..##...##..##................##..###...###.##....##..##.....##....##.....##..##.......##....##
## .##..........##....##.....##..##..##..##...##................##..####.####.##........##.....##....##.....##..##.......##......
## ..######.....##....########...##..#####....######............##..##.###.##.##...####.##.....##....##.....##..##........######.
## .......##....##....##...##....##..##..##...##................##..##.....##.##....##..##.....##....##.....##..##.............##
## .##....##....##....##....##...##..##...##..##................##..##.....##.##....##..##.....##....##.....##..##.......##....##
## ..######.....##....##.....##.####.##....##.########.#######.####.##.....##..######....#######.....##....####.########..######.

class Strike_ImgUtils:

    @staticmethod    
    def np_gsk1(size=[200,200], m=[100,100], S=[100,100], gain=1, rotation=0):
        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        x,y = m
        sx,sy = size
        valx = gaussian(np.linspace(0,sx,sx),x,S[0]) * gain
        valy = gaussian(np.linspace(0,sy,sy),y,S[1]) * gain
        kernel = np.einsum('i,j->ij', valx, valy)
        return kernel

    @staticmethod
    def np_gsk2(size=[200,200], m=[100,100], S=[100,100], gain=1, rotation=0, dtype=np.float):        
        #### REMOVE THIS #####
        # m = np.random.normal(m,[50,50],2)
        # S = np.random.normal(S,[10,10],2)
        ######################        
        if dtype != float and dtype != np.float:
            gain *= float(np.iinfo(dtype).max)
        if np.shape(S) == (2,):
            S = np.array([[S[0],0],[0,S[1]]],dtype=np.float)
        S = np.linalg.inv(S)
        if (rotation != 0):
            sr = np.sin(float(rotation))
            cr = np.cos(float(rotation))
            T  = np.array([[cr,-sr],[sr,cr]])
            S  = np.dot(np.dot(T.transpose(), S), T)    
        img = np.zeros(size,dtype=dtype)
        for x in range(0,size[0]):
            for y in range(0,size[1]):
                xy = [x,y] - np.array(m)
                fp = np.exp( -np.dot(np.dot(xy,S),xy) / 2 ) * gain
                img[x,y] = np.cast[dtype](fp)
        return img

    @staticmethod
    def np_gsk3(size=[200,200], m=[100,100], S=[100,100], gain=1, rotation=0):
        def gaussian(xy, IS):
            x,y = np.unravel_index(xy,size)
            xy  = np.array([x,y]) - m
            return np.exp( -np.dot(np.dot(xy,IS),xy)/2 ) * gain
        sizex, sizey = size
        if np.shape(S) == (2,):
            S = np.array([[S[0],0],[0,S[1]]],dtype=np.float)
        S = np.linalg.inv(S)
        if (rotation > 0):
            sr = np.sin(rotation)
            cr = np.cos(rotation)
            T  = np.array([[cr,-sr],[sr,cr]])
            S  = np.dot(np.dot(T.transpose(), S), T)
        img = np.array([gaussian(i,m,S,gain) for i in range(sizex*sizey)]).reshape(sizex,sizey)
        return img


    @staticmethod
    def tf_gsk2(size=[200,200], m=[100,100], S=[100,100], gain=1, rotation=0, dtype=tf.float32):
        def add_gauss(img,x,y):
            # m = tf.convert_to_tensor(m, dtype=tf.float32)
            # S = tf.convert_to_tensor(S,dtype=tf.float32)
            xy = [x,y] - m
            return img + tf.exp( - tf.tensordot(tf.tensordot(xy,S,1), xy,1) / 2 ) * gain

        def add_gauss_Y(img,y):
            x = tf.range(tf.cast(tf.shape(img)[0],dtype), dtype=dtype)
            return tf.map_fn(lambda l: add_gauss(l[0],l[1],y),(img,x),dtype=dtype)

        def proc_img(img):
            y = tf.range(tf.cast(tf.shape(img)[0],dtype), dtype=dtype)
            return tf.map_fn(lambda l: add_gauss_Y(l[0],l[1]),(img,y),dtype=dtype)        

        # m = tf.convert_to_tensor(m, dtype=tf.float32)
        # S = tf.convert_to_tensor(S,dtype=tf.float32)
        gain     = tf.convert_to_tensor(gain, dtype=dtype)
        rotation = tf.convert_to_tensor(rotation, dtype=dtype)
        m = tf.random_normal((2,),m,[50,50])
        S = tf.random_normal((2,),S,[10,10])
        if S.shape == (2,):
            S = tf.convert_to_tensor([[S[0],0],[0,S[1]]],dtype=dtype)
        S = tf.linalg.inv(S)
        if (rotation != 0):
            sr = tf.sin(rotation)
            cr = tf.cos(rotation)
            T  = tf.convert_to_tensor([[cr,-sr],[sr,cr]])
            S  = tf.matmul(tf.matmul(T, S, transpose_a=True), T)
        img = tf.zeros(size,dtype=dtype)
        return proc_img(img)


    @staticmethod
    def benchmark_np_image_gen(function, size=[200,200], number_of_images=64):
        static = Strike_ImgUtils.benchmark_np_image_gen
        def gen():
            import itertools
            for i in itertools.count(1):
                yield (i,function(size=size))
        static.sess  = tf.Session() # tf.Session(config=tf.ConfigProto(log_device_placement=True))
        start_time = time.time()
        dataset  = tf.data.Dataset.from_generator(gen, output_shapes=((),tuple(size)), output_types=(tf.int32, tf.float32) )
        dataset.batch(number_of_images)
        dataset.prefetch(number_of_images)        
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        imgs = []
        for i in range(number_of_images):
            static.sess.run(iterator.initializer)
            label,img = static.sess.run(next_element)
            imgs.append(img)
        return time.time() - start_time, imgs

    @staticmethod
    def benchmark_tf_image_gen(function, size=[200,200], number_of_images=64):
        session    = tf.Session() # tf.Session(config=tf.ConfigProto(log_device_placement=True))        
        start_time = time.time()
        dataset    = tf.data.Dataset.range(number_of_images).map(lambda x: function(size=size))
        dataset.batch(number_of_images)
        dataset.prefetch(number_of_images)        
        iterator   = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        imgs = []
        for i in range(number_of_images):
            session.run(iterator.initializer)
            img = session.run(next_element)
            imgs.append(img)
        return time.time() - start_time, imgs

    @staticmethod
    def example_save_dataset_to_mdsplus(batch_size=100):
        import MDSplus as mds
        import os
        os.environ["striked1_path"] = "."
        tree = mds.tree.Tree('striked1',2,'NEW')
        node = tree.addNode('gsk2','SIGNAL')
        
        dataset  = tf.data.Dataset.range(batch_size).map(lambda x: Strike_ImgUtils.np_gsk2())        
        dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        el = iterator.get_next()

        sess = tf.Session()
        imgs = []
        for i in range(batch_size):
            sess.run(iterator.initializer)
            img = sess.run(el)
            img = tf.image.convert_image_dtype(img,dtype=tf.uint8)
            img = tf.reshape(img,(200,200,1))
            jpg = tf.image.encode_jpeg(img, format='grayscale')
            jpgb = tf.decode_raw(jpg,out_type=tf.uint8)
            opq = mds.mdsarray.Uint8Array(jpgb.eval())
            pos = mds.mdsscalar.Float32(i)
            node.makeSegment(pos,pos,mds.mdsarray.Int64Array([len(opq)]),opq)
            imgs.append(img)
        tree.write()
        tree.close()


    @staticmethod
    def example_read_first_image_from_mdsplus():
        import MDSplus as mds
        import os
        os.environ['striked1_path'] = '.'
        tree = mds.tree.Tree('striked1',2,'EDIT')
        node = tree.getNode('gsk2')
        jpg  = np.array(node.getSegment(0).data())
        img = tf.image.decode_jpeg(jpg.tostring())
        img = tf.reshape(img, (200,200))
        return img



class Strike_DatasetGsk2():

    def __init__(self, size, media_type='disk', *args, **kwargs ):
        self.enum_mediaTypes = ['disk', 'mdsplus', 'zipfile']
        self.dataset_size = 100
        self.batch_size   = 1
        self.img_size     = [200,200]

        super(Strike_DatasetGsk2, self).__init__()

    def _init_generator(self, generator):
        def gen():
            import itertools
            for i in itertools.count(1):
                yield (i,generator(size=self.img_size))
        
        shape = ((),tuple(self.img_size))
        types = ((tf.int32,tf.float32), tf.float32)
        return tf.data.Dataset.from_generator(generator=gen, output_shapes=shape, output_types=types)

                
    def set_batch(self, batch_size=1):
        
        pass

    def write_to_mdsplus(self, tree, pulse, node):
        pass
    
    def write_to_disk(self, path, filename):
        pass

    def write_to_zipfile(self, filename):
        pass






# ////////////////////////////////////////////////////////////////////////////////////////////////////// #
# /// IMG GENERATOR BASE /////////////////////////////////////////////////////////////////////////////// #
# ////////////////////////////////////////////////////////////////////////////////////////////////////// #

"""
.####.##.....##....###.....######...########..######...########.##....##.########.########.....###....########..#######..########.
..##..###...###...##.##...##....##..##.......##....##..##.......###...##.##.......##.....##...##.##......##....##.....##.##.....##
..##..####.####..##...##..##........##.......##........##.......####..##.##.......##.....##..##...##.....##....##.....##.##.....##
..##..##.###.##.##.....##.##...####.######...##...####.######...##.##.##.######...########..##.....##....##....##.....##.########.
..##..##.....##.#########.##....##..##.......##....##..##.......##..####.##.......##...##...#########....##....##.....##.##...##..
..##..##.....##.##.....##.##....##..##.......##....##..##.......##...###.##.......##....##..##.....##....##....##.....##.##....##.
.####.##.....##.##.....##..######...########..######...########.##....##.########.##.....##.##.....##....##.....#######..##.....##
"""

class ImageGenerator():
    ''' Base class to handle Strike Image generators input output to file and to mdsplus trees
    '''
    __metaclass__ = abc.ABCMeta
    class CfgDict(dict):
        ''' A class to overload a dictionary with direct access to keys as internal methods
        '''
        def __init__(self, *args, **kwargs):
            super(ImageGenerator.CfgDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
        
        def __add__(self, other):
            super(ImageGenerator.CfgDict, self).update(other)
            return self

    def __init__(self, *args, **kwargs):
        self.cfg = ImageGenerator.CfgDict({
            'dataset_size' : 36000,
            'img_size'     : [200,300],
            'batch_size'   : 36,
            'tree_path'    : '.',
            'n_threads'    : 8,
        })
        self.cfg.update(kwargs)
        self.dataset = 0
        self.batch   = 0

    def get_config(self):
        ''' Get configuration dictionary
        '''
        return self.cfg

    def get_dataset(self):
        ''' Get current dataset or initialize the internal generation dataset if 
            any source was previously selected
        '''
        if self.dataset == 0:
            self.read_from_generator()
        return self.dataset


    @abc.abstractmethod
    def get_name(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_data(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape(self):
        # x,img = self.get_data()
        # return (np.shape(x), np.shape(img))
        raise NotImplementedError

    @abc.abstractmethod
    def get_types(self):
        raise NotImplementedError

    # iterable
    def __len__(self):
        return self.cfg.dataset_size

    # iterable
    def __getitem__(self, item):
        return self.get_data()

    @staticmethod
    def to_list(model):
        mlist = []
        if isinstance(model,list):
            for el in model:
                mlist += ImageGenerator.to_list(el)
        elif isinstance(model,dict) or isinstance(model,OrderedDict):
            for el in model.values():
                mlist += ImageGenerator.to_list(el)
        # elif callable(getattr(model, "tolist", None)):
        #     mlist += model.tolist()
        else:
            mlist.append(model)
        return mlist

    @staticmethod
    def compute_model_size(model):
        return len(ImageGenerator.to_list(model))


    def compute_stat(self, size=0):    
        import copy
        llen = ImageGenerator.compute_model_size(self.get_model())
        st_l = [copy.deepcopy(Stat2()) for i in range(llen)]        
        # st_x = Stat2()
        if size == 0: size = self.cfg.dataset_size
        with tf.Session() as sess:
            it = self.dataset.make_one_shot_iterator()
            ne = it.get_next()
            for i in range(size):
                # if i%100 == 0: print(i)
                lab, img = sess.run(ne)
                for j in range(llen):
                    st_l[j] += lab[j]
        return [st.mean() for st in st_l], [st.rms() for st in st_l]

    def compute_minmax(self, size=0):    
        import copy
        llen = ImageGenerator.compute_model_size(self.get_model())
        st_l = [copy.deepcopy(Stat2()) for i in range(llen)]        
        # st_x = Stat2()
        if size == 0: size = self.cfg.dataset_size
        with tf.Session() as sess:
            it = self.dataset.make_one_shot_iterator()
            ne = it.get_next()
            for i in range(size):
                # if i%100 == 0: print(i)
                lab, img = sess.run(ne)
                for j in range(llen):
                    st_l[j] += lab[j]
        return [st.min() for st in st_l], [st.max() for st in st_l]

    ## /////////////////////////////////////////////////////////////////////////////////// ##
    ## //  READ WRITE  TO MEDIA  ///////////////////////////////////////////////////////// ##

    def read_from_generator(self):
        ''' Set the dataset to read from inline generated function that collate image and labels
            form the abstract get_data() function.
        '''
        def gen():
            import itertools
            for i in itertools.count(1):
                l,img = self.get_data()
                yield (l,img)
        cfg = self.cfg
        self.dataset = tf.data.Dataset.from_generator(gen, self.get_types(), self.get_shape())
        self.batch   = self.dataset.batch(cfg.batch_size)
        return self.dataset

    # NOT WORKING
    def read_from_generator2(self):
        def gen_static():
            print('static data generated')
            return self.get_data()
        ImageGenerator.read_from_generator2.dummy = gen_static()
        def py_func_decorator(output_types=None, output_shapes=None, stateful=True, name=None):
            def decorator(func):
                def call(*args, **kwargs):
                    return tf.py_function(func = func, inp=[], Tout=output_types) 
                return call
            return decorator

        def from_indexable(self, output_types, output_shapes=None, num_parallel_calls=None, stateful=True, name=None):
            ds = tf.data.Dataset.range(len(self))
            @py_func_decorator(output_types, output_shapes, stateful=stateful, name=name)
            def index_to_entry():
                print('doing')
                lab, img = self.get_data()
                print('done')
                return (lab, img)
            return ds.map(index_to_entry, num_parallel_calls=num_parallel_calls)
        
        types = (tf.float64, tf.uint8)
        ds = from_indexable(self, 
                            output_types=types,
                            num_parallel_calls=6) #self.cfg.n_threads)
        self.dataset = ds                            
        self.batch = ds.batch(self.cfg.batch_size)
        return ds


    def write_to_mds(self, tree_name, tree_pulse):
        import MDSplus as mds
        import os, sys, pickle
        cfg = self.cfg
        os.environ[tree_name+"_path"] = cfg.tree_path
        tree = mds.tree.Tree(tree_name,tree_pulse,'NEW')
        n_sup = tree.addNode('\\top.'+self.get_name())
        tree.setDefault(n_sup)
        n_cfg = tree.addNode('cfg')                
        n_lab = tree.addNode('labels','SIGNAL')
        n_img = tree.addNode('img','SIGNAL')
        
        # // WRITE LOOP //
        n_cfg.putData(cfg)
        it = self.dataset.make_one_shot_iterator()
        el = it.get_next()
        with tf.Session() as sess:
            size = cfg.dataset_size
            for i in range(size):
                if i%int(size/1000) == 0: Strike_PlotUtils.printProgressBar(i,size,'completed')
                lab, img = sess.run(el)
                img_shape = np.shape(img)
                img = tf.image.convert_image_dtype(img,dtype=tf.uint8)
                img = tf.reshape(img, img_shape+(1,))
                jpg = tf.image.encode_jpeg(img, format='grayscale')
                jbn = tf.decode_raw(jpg,out_type=tf.uint8)                    
                opq = mds.mdsarray.Uint8Array(sess.run(jbn))
                pos = mds.mdsscalar.Float32(i)
                n_img.makeSegment(pos,pos,mds.mdsarray.Int64Array([len(opq)]),opq)
                lbn = mds.mdsarray.Uint8Array(bytearray(pickle.dumps(lab)))
                n_lab.makeSegment(pos,pos,mds.mdsarray.Int64Array([sys.getsizeof(lbn)]),lbn)
        tree.write()
        tree.close()

    def read_from_mds(self, tree_name, tree_pulse):
        import MDSplus as mds
        import os,sys, pickle, PIL, io        
        def gen():
            import itertools
            for i in itertools.count(1):
                tree = self.tree
                lab = tree.getNode('labels').getSegment(i).data()
                lab = pickle.loads(lab)
                jpg = np.array(tree.getNode('img').getSegment(i).data())
                img = PIL.Image.open(io.BytesIO(jpg))
                yield lab,imgopen

        cfg = self.cfg
        os.environ[tree_name+'_path'] = cfg.tree_path
        self.tree = mds.tree.Tree(tree_name, tree_pulse, 'NORMAL')
        self.tree.setDefault(self.tree.getNode(self.get_name()))

        # build dataset from file read using a generation funciton
        self.dataset = tf.data.Dataset.from_generator(gen, self.get_types(), self.get_shape() )
        self.batch = self.dataset.batch(cfg.batch_size)
        return self.dataset


    def write_to_files(self, file_name):
        import os, csv, json
        from PIL import Image        
        cfg = self.cfg        
        cwd = os.getcwd()                
        os.chdir(cfg.tree_path)        
        try: os.mkdir(self.get_name())
        except: pass
        os.chdir(self.get_name())

        # save configuration
        with open(file_name+'_cfg.json', 'w') as f:
            json.dump(cfg, f)

        # input from dataset
        ds = self.dataset
        it = ds.make_one_shot_iterator()
        el = it.get_next()

        # save labels into csv file                
        with   tf.Session() as sess, \
               open(file_name+'.csv', "w") as f:
            writer = csv.writer (f)
            size = cfg.dataset_size
            for i in range(size):
                if i%int(size/1000) == 0: Strike_PlotUtils.printProgressBar(i,size,'completed')
                lab, img = sess.run(el)
                pil = Image.fromarray(img)
                name = file_name+'_'+str(i)+'.jpg'
                pil.save(name,'JPEG')
                writer.writerow(lab)
                f.flush()
        # go back to original cwd
        os.chdir(cwd)

    def read_from_files(self, file_name):
        import os, csv, json
        from PIL import Image
        cfg = self.cfg        
        os.chdir(cfg.tree_path)
        os.chdir(self.get_name())
        
        # restore configuration
        with open(file_name+'_cfg.json', 'r') as f:
            cfg = ImageGenerator.CfgDict(json.load(f))
        labels = []
        with open(file_name+'.csv','r') as f:
            reader = csv.reader(f)
            for row in reader:
                labels.append(row)
        def gen():
            import itertools
            for i in itertools.count(0):
                lab = labels[i]
                img = Image.open(file_name+'_'+str(i)+'.jpg')
                yield lab,img
        self.dataset = tf.data.Dataset.from_generator(gen, self.get_types(), self.get_shape())
        self.batch = self.dataset.batch(cfg.batch_size)
        return self.dataset



# ///////////////////////////////////////////////////////////////////////////////////////////// #
# ///  GENERATORS  //////////////////////////////////////////////////////////////////////////// #
# ///////////////////////////////////////////////////////////////////////////////////////////// #

"""
.##.....##.##....##..######...########.##....##
.###...###..##..##..##....##..##.......###...##
.####.####...####...##........##.......####..##
.##.###.##....##....##...####.######...##.##.##
.##.....##....##....##....##..##.......##..####
.##.....##....##....##....##..##.......##...###
.##.....##....##.....######...########.##....##
"""

class MyGen(ImageGenerator):
    ''' Image generator example using gaussian source shaped spots
        generate a number of spaced gauss shapes along Y axes 
    '''
    __metaclass__ = abc.ABCMeta
    model_name = 'MyGen'

    def __init__(self, *args, **kwargs):
        self.dtype = tf.float32
        super(MyGen, self).__init__(*args, **kwargs)
        
        ## internal configuration dictionary
        self.cfg += {
            'MyGen_ver'  : 1,
            'n_gauss'    : 2,
            'gen_spread' : [30,30],    # position spread variance
            'gen_size'   : [400,120],  # size m s normal for x and y axes
            'gen_rotz'   : [0,  180],  # uniform rotation angle min/max
            'gen_gain'   : [1,0.1],
        }
        self.cfg.update(kwargs)
        model_len = ImageGenerator.compute_model_size(self.get_model()) 
        self.shapes = (model_len), (tuple(self.cfg.img_size))
        self.types  = np.float, np.uint8
        
    @abc.abstractmethod
    def get_name(self):
        return self.model_name

    @abc.abstractmethod
    def get_data(self):
        model = self.get_model()        
        cfg = self.cfg
        id_pos = 0
        img = np.zeros(cfg.img_size, dtype=float)
        for cl in model:            
            id_pos += 1
            xy = [cfg.img_size[0]/2, cfg.img_size[1]/(cfg.n_gauss+1) * id_pos]
            cl['pos'][0:2] = np.random.normal(xy,cfg.gen_spread,2)
            cl['std'][0:2] = np.abs( np.random.normal(cfg.gen_size[0],cfg.gen_size[1],2) )
            cl['std'][2:3] = np.random.uniform(cfg.gen_rotz[0],cfg.gen_rotz[1],1)  / 180 * np.pi
            # cl['gain']     = 0.5 #np.absolute( np.random.normal(cfg.gen_gain[0],cfg.gen_gain[1]) )
            img += Strike_ImgUtils.np_gsk2(cfg.img_size, cl['pos'], cl['std'][0:2], rotation=cl['std'][2], gain=0.5)
        img = tf.Session().run(tf.image.convert_image_dtype(img, dtype=tf.uint8))
        return (ImageGenerator.to_list(model), img)

    @abc.abstractmethod
    def get_model(self):
        import copy
        model = OrderedDict([
            ('pos', [1,2]),
            ('std', [3,4,5]),
            # ('gain',6),
        ])
        return [copy.deepcopy(model) for i in range(self.cfg.n_gauss)]
    

    @abc.abstractmethod
    def get_shape(self):
        return self.shapes

    @abc.abstractmethod
    def get_types(self):        
        return self.types


"""
.########.####.##....##..######..##.....##.##.....##..######...########.##....##
.##........##..###...##.##....##.##.....##.###...###.##....##..##.......###...##
.##........##..####..##.##.......##.....##.####.####.##........##.......####..##
.######....##..##.##.##..######..##.....##.##.###.##.##...####.######...##.##.##
.##........##..##..####.......##.##.....##.##.....##.##....##..##.......##..####
.##........##..##...###.##....##.##.....##.##.....##.##....##..##.......##...###
.########.####.##....##..######...#######..##.....##..######...########.##....##
"""

class EinsumGen(ImageGenerator):
    ''' Image generator example using gaussian source shaped spots
        generate a number of spaced gauss shapes along Y axes 
    '''
    __metaclass__ = abc.ABCMeta
    model_name = 'EinsumGen'

    class Model(Struct):
        pos = [1.,2.]
        std = [3.,4.]

        def tolist(self):
            # lout = np.concatenate([self.pos, self.std],0)
            return self.pos + self.std

    def __init__(self, *args, **kwargs):
        self.dtype = tf.float32
        super(EinsumGen, self).__init__(*args, **kwargs)        
        ## internal configuration dictionary
        self.cfg += {
            'MyGen_ver'  : 1,
            'n_gauss'    : 1,
            'gen_spread' : [30,30],    # position spread variance
            'gen_size'   : [20,6],    # size m s normal for x and y axes
            'gen_rotz'   : [0,  180],  # uniform rotation angle min/max
            'gen_gain'   : [1,0.1],
        }
        self.cfg.update(kwargs)
        model_len = ImageGenerator.compute_model_size(self.get_model())
        self.shapes = (model_len), (tuple(self.cfg.img_size))
        self.types  = np.float, np.uint8
        
    @abc.abstractmethod
    def get_name(self):
        return self.model_name

    @abc.abstractmethod
    def get_data(self):
        model = self.get_model()
        cfg = self.cfg
        id_pos = 0
        img = np.zeros(cfg.img_size, dtype=float)
        for cl in model:            
            id_pos += 1
            xy = [cfg.img_size[0]/2, cfg.img_size[1]/(cfg.n_gauss+1) * id_pos]            
            cl.pos = np.random.normal(xy,cfg.gen_spread,2).tolist()
            cl.std = np.abs( np.random.normal(cfg.gen_size[0],cfg.gen_size[1],2) ).tolist()
            img += Strike_ImgUtils.np_gsk1(cfg.img_size, cl.pos, cl.std, rotation=0, gain=1/cfg.n_gauss)
        img = tf.Session().run(tf.image.convert_image_dtype(img, dtype=tf.uint8))
        return (ImageGenerator.to_list(model), img)

    @abc.abstractmethod
    def get_model(self):
        return [EinsumGen.Model() for i in range(self.cfg.n_gauss)]
    
    @abc.abstractmethod
    def get_shape(self):
        return self.shapes

    @abc.abstractmethod
    def get_types(self):        
        return self.types





"""
.##.....##.##....##.####..######...########.##....##
.##.....##.###...##..##..##....##..##.......###...##
.##.....##.####..##..##..##........##.......####..##
.##.....##.##.##.##..##..##...####.######...##.##.##
.##.....##.##..####..##..##....##..##.......##..####
.##.....##.##...###..##..##....##..##.......##...###
..#######..##....##.####..######...########.##....##
"""

class UniGen(ImageGenerator):
    ''' Image generator example using gaussian source shaped spots
        generate a number of spaced gauss shapes along Y axes 
    '''
    __metaclass__ = abc.ABCMeta
    model_name = 'UniGen'

    def __init__(self, *args, **kwargs):
        self.dtype = tf.float32
        super(UniGen, self).__init__(*args, **kwargs)
        
        ## internal configuration dictionary
        self.cfg += {
            'version'    : 1,
            'n_gauss'    : 1,
            'gen_size'   : [400,120],  # size m s normal for x and y axes
            'gen_rotz'   : [0,  90],  # uniform rotation angle min/max
            'gen_gain'   : [1,0.1],
        }
        self.cfg.update(kwargs)
        model_len = ImageGenerator.compute_model_size(self.get_model()) 
        self.shapes = (model_len), (tuple(self.cfg.img_size))
        self.types  = np.float, np.uint8
        

    @abc.abstractmethod
    def get_name(self):
        return self.model_name

    @abc.abstractmethod
    def get_data(self):
        model = self.get_model()        
        cfg = self.cfg
        id_pos = 0
        img = np.zeros(cfg.img_size, dtype=float)
        for cl in model:            
            dx = cfg.img_size[1]/cfg.n_gauss * np.array([id_pos, id_pos+1])
            id_pos += 1            
            cl['pos'][0:2] = np.random.uniform([0,dx[0]],[cfg.img_size[0],dx[1]],2)
            cl['std'][0:2] = np.abs( np.random.normal(cfg.gen_size[0],cfg.gen_size[1],2) )
            cl['std'][2:3] = np.random.uniform(cfg.gen_rotz[0],cfg.gen_rotz[1],1)  / 180 * np.pi
            img += Strike_ImgUtils.np_gsk2(cfg.img_size, cl['pos'], cl['std'][0:2], rotation=cl['std'][2], gain=0.5)
        img = tf.Session().run(tf.image.convert_image_dtype(img, dtype=tf.uint8))
        return (ImageGenerator.to_list(model), img)

    @abc.abstractmethod
    def get_model(self):
        import copy
        model = OrderedDict([
            ('pos', [1,2]),
            ('std', [3,4,5]),
        ])
        return [copy.deepcopy(model) for i in range(self.cfg.n_gauss)]
    

    @abc.abstractmethod
    def get_shape(self):
        return self.shapes

    @abc.abstractmethod
    def get_types(self):        
        return self.types








## DIRECT TEST ##
# 
if __name__ == '__main__':
    # test new model
    # Stat2_test()
    ug = UniGen(n_gauss=1)
    ds = ug.read_from_generator().batch(9)
    it = ds.make_one_shot_iterator()
    el = it.get_next()
    with tf.Session() as sess:
        lab, X = sess.run(el)
        Strike_PlotUtils.show_images(X, xys3=lab)
        plt.show()
    # ug.cfg.dataset_size = 20000
    # ug.write_to_files('uni1')
    













