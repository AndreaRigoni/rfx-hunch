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
.##.....##..#######..########..########.##......
.###...###.##.....##.##.....##.##.......##......
.####.####.##.....##.##.....##.##.......##......
.##.###.##.##.....##.##.....##.######...##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##..#######..########..########.########
"""

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline


class SPFIT(tf.keras.Model):

  def __init__(self, feature_dim=2000, latent_dim=20):
    super(SPFIT, self).__init__()
    self.latent_dim = latent_dim
    self.feature_dim = feature_dim    
    self.inference_net = tf.keras.Sequential(
      [
          tf.keras.layers.Input(shape=(feature_dim,)),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Reshape(target_shape=(feature_dim,1)),
          tf.keras.layers.Conv1D(kernel_size=(int(feature_dim/latent_dim)), filters = 20, activation='relu'),
          tf.keras.layers.Dense(100),
          tf.keras.layers.Dense(latent_dim + latent_dim),
      ]
    )
  
  def get_metrics(self,X,s):
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    def fn1(x,y,s):
      y_ = tf.map_fn(lambda x: s[self._x_id(x)], x)
      r2  = r2_score(y,y_)
      mse = mean_squared_error(y,y_)
      return tf.convert_to_tensor([r2,mse], dtype=tf.float32)
    def wrap_fn1(t):
      X,s = t[0], t[1]
      x,y,att = X
      return fn1(tf.boolean_mask(x,att),tf.boolean_mask(y,att),s)
    met = tf.map_fn(lambda x: wrap_fn1(x), (X,s), dtype=(tf.float32))
    return met

  def smooth_usp(self, X):
    from scipy import signal
    x,y,att = X
    y_max = np.nanmax(y)
    def fn(X):
      x,y,att = X[0],X[1]/y_max,X[2]
      w = signal.tukey(np.sum(att))
      k = UnivariateSpline(tf.boolean_mask(x,att),tf.boolean_mask(y,att), w)
      k.set_smoothing_factor(0.005)
      return tf.convert_to_tensor(k(self._x), dtype=tf.float32)
    s = tf.map_fn(lambda x: fn(x), X, dtype=tf.float32)
    return s

  def smooth_rusp(self, X, k=3, s=0.005):
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from scipy import signal
    y_max = np.nanmax(X[1])
    def repeat(x,r):
      return tf.reshape(tf.tile(x,[r]),[r,-1])
    def point_reject_score(x,y,w=None):
      if w is None:
        w = tf.eye(len(y))
        w = tf.ones_like(w) - w             
      else:
        w = repeat(w,len(y))
        w -= tf.eye(len(y))
      sp = [ UnivariateSpline(x, y, row, k=k, s=s ) for row in w ]
      r2 = [ tf.abs(r2_score(y,sp(x))) for sp in sp ]
      return tf.convert_to_tensor(w[tf.argmax(r2)], dtype=tf.float32)
    def fn(X):
      x,y,a = X[0],X[1]/y_max,X[2]
      x = tf.boolean_mask(x,a)
      y = tf.boolean_mask(y,a)      
      w = tf.convert_to_tensor(signal.tukey(np.sum(a)), dtype=tf.float32)
      w = point_reject_score(x,y,w)
      sp = UnivariateSpline(x, y, w, k=k, s=s)
      return tf.convert_to_tensor(sp(self._x), dtype=tf.float32)
    s = tf.map_fn(lambda x: fn(x), X, dtype=tf.float32)
    return s

  def smooth_krr(self, X):
    from sklearn.kernel_ridge import KernelRidge
    # from sklearn.model_selection import GridSearchCV
    _,y,_ = X
    y_max = np.nanmax(y)
    def fn(X):
      x,y,a = X[0],X[1]/y_max,X[2]
      x = tf.reshape(tf.boolean_mask(x,a), [-1,1])
      y = tf.reshape(tf.boolean_mask(y,a), [-1,1])
      # k = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
      k = KernelRidge(alpha=0.00001, kernel='polynomial',degree=5, gamma=0.8)
      k.fit(x,y)
      return tf.convert_to_tensor(k.predict(self._x.reshape(-1,1)), dtype=tf.float32)
    s = tf.map_fn(lambda x: fn(x), X, dtype=tf.float32)
    return s

  def smooth_gpr(self, X):
    from sklearn.gaussian_process import GaussianProcessRegressor
    _,y,_ = X
    y_max = np.nanmax(y)
    def fn(X):
      x,y,a = X[0],X[1]/y_max,X[2]
      x = tf.reshape(tf.boolean_mask(x,a), [-1,1])
      y = tf.reshape(tf.boolean_mask(y,a), [-1,1])
      k = GaussianProcessRegressor(alpha=0.000001)
      k.fit(x,y)
      return tf.convert_to_tensor(k.predict(self._x.reshape(-1,1)), dtype=tf.float32)
    s = tf.map_fn(lambda x: fn(x), X, dtype=tf.float32)
    return s

  def smooth_knn(self, X):
    from sklearn.neighbors import KNeighborsRegressor
    _,y,_ = X
    y_max = np.nanmax(y)
    def fn(X):
      x,y,a = X[0],X[1]/y_max,X[2]
      x = tf.reshape(tf.boolean_mask(x,a), [-1,1])
      y = tf.reshape(tf.boolean_mask(y,a), [-1,1])
      k = KNeighborsRegressor(n_neighbors=3, weights='distance')
      k.fit(x,y)
      return tf.convert_to_tensor(k.predict(self._x.reshape(-1,1)), dtype=tf.float32)
    s = tf.map_fn(lambda x: fn(x), X, dtype=tf.float32)
    return s

  def smooth_svm(self, X):
    from sklearn.svm import SVR
    _,y,_ = X
    y_max = np.nanmax(y)
    def fn(X):
      x,y,a = X[0],X[1]/y_max,X[2]
      x = tf.reshape(tf.boolean_mask(x,a), [-1,1])
      y = tf.reshape(tf.boolean_mask(y,a), [-1,1])
      k = SVR(C=10000.0, cache_size=200, coef0=2.0, degree=5, epsilon=0.0001,
              gamma='auto_deprecated', kernel='rbf', max_iter=-1, shrinking=True,
              tol=0.1, verbose=False)
      k.fit(x,y)
      return tf.convert_to_tensor(k.predict(self._x.reshape(-1,1)), dtype=tf.float32)
    s = tf.map_fn(lambda x: fn(x), X, dtype=tf.float32)
    return s

  def set_dataset(self,ds):
    import copy
    self._ds = copy.deepcopy(ds)
    self._ds.rebalance_prel()
    self._ds.set_null()
    _min = np.nanmin(self._ds['prel'])
    _max = np.nanmax(self._ds['prel'])
    self._x = np.linspace(_min,_max, self.feature_dim)
    self._x_id = lambda x: int((x+_min)/(_max-_min)*self.feature_dim)
  
  def get_tf_dataset(self):
    return self._ds.get_tf_dataset()
   
    
  def show_batch(self, size=100, pause=0.05):
      data = self.get_tf_dataset().batch(size)
      X = data.make_one_shot_iterator().get_next()
      x,y,att = X
      y_max = np.nanmax(y)
      # s = self.smooth_rusp(X)
      s1 = self.smooth_krr(X)
      s2 = self.smooth_gpr(X)
      s3 = self.smooth_knn(X)
      s4 = self.smooth_svm(X)
      plt.ion()
      fig = plt.figure('usp')
      for i in range(len(s1)):
        plt.clf()
        plt.plot(X[0][i],X[1][i]/y_max,'.')
        plt.plot(self._x, s1[i], label='krr')
        plt.plot(self._x, s2[i], label='gpr')
        plt.plot(self._x, s3[i], label='knn')
        plt.plot(self._x, s4[i], label='svm')
        plt.legend()
        plt.pause(pause)

