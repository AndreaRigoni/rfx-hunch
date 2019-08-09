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
import os  


qsh1 = Htls.QSH_Dataset()
qsh1.load('te_db_1.npy')

qsh = Htls.QSH_Dataset()
qsh.dim = 15
qsh.load(ipysh.abs_builddir+'/te_db_r15_clean.npy')
qsh.set_null(np.nan)



