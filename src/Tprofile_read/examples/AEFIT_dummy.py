
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
import models.AEFIT as aefit

ipysh.Bootstrap_support.debug()

ds = dummy.Dummy_g1data()
m = aefit.AEFIT(latent_dim=2)

# aefit.test_dummy(m)
print('all conifgured for the test')
print('now type: aefit.test_dummy(m)')

