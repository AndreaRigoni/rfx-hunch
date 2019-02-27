import matplotlib.pyplot as plt
import numpy as np

import sys 
import MDSplus as mds

plt.ion()

qd = np.load( 'te_db_1.npy' )

q_rho = qd[:]['rho']
q_te = qd[:]['te']
q_prel = qd[:]['prel']
q_label = qd[:]['label']

print qd.dtype.names

fig = plt.figure( 'te' )
fig.set_size_inches( 8, 5.25 )
fig.clf()

ax = plt.gca()
# ax.plot( q_rho, q_te, ',' )
ax.plot( q_prel, q_rho, ',' )
ax.set_xlim( -0.45, 0.45 )

kk = np.logical_and( q_prel > 0.07, q_prel < 0.08 )
casi_strani = np.where( kk )[0]
print q_label[casi_strani]

