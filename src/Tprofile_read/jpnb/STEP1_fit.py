

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
import models.AEFIT1 as aefit




ds = Hunch.QSH_Dataset()
ds.loadData_npy(ipysh.abs_builddir+'/te_db_1.npy')

# ae = aefit.AEFIT1(feature_dim=100, latent_dim=20)
# ae.set_dataset(ds)




