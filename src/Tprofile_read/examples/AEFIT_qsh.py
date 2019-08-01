from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors 

import ipysh
import Hunch_utils  as Htls
import Hunch_lsplot as Hplt

import Dummy_g1data as dummy
import models.AEFIT0 as aefit

# ipysh.Bootstrap_support.debug()

qsh = Htls.QSH_Dataset()

import os  
# if os.path.isfile('te_db_2.npy'):    # True  
#     qsh.load('te_db_2.npy')
# else:
#     qsh.load('te_db_1.npy')
#     qsh.set_null(np.nan)
#     qsh.rebalance_prel()
#     qsh.save('te_db_2.npy')

qsh.load('te_db_1.npy')




