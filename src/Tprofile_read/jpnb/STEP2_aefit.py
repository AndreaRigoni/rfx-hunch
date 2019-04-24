
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
import Dummy_g1data as dummy
import models.AEFIT as aefit



ds = dummy.Dummy_g1data()
m = aefit.AEFIT(latent_dim=2)


# aefit.test_dummy(m)
print('all conifgured for the test')
print('now type this: aefit.test_dummy(m)')



