import matplotlib.pyplot as plt
from scipy.io import readsav
import numpy as np

import sys 
import MDSplus as mds

data_dir = '/scratch/gobbin/rigoni/'
argv = sys.argv 

if len( argv ) > 1 :
	try:
		shot = int( argv[1] )
	except:
		print "invalid shot: ", argv[1]
		sys.exit(0)
else:
	shot = 30808
	
if ( shot < 15600 or shot > 39391 ) :
	print "invalid shot num: ", shot
	sys.exit(0)


file = 'dsx3_%d.sav' % shot
print file

try:
	x = readsav( data_dir+file, python_dict=False ).st
	# x = readsav( data_dir+file, python_dict=True )
except:
	print "invalid file: ", file
	sys.exit(0)

plt.ion()

fig = plt.figure( 'i_pla' )
fig.set_size_inches( 8, 5.25 )
fig.clf()
ax = plt.gca()
ax.plot( x.t[0], x.ip[0] )
ax.set_title( shot )

fig = plt.figure( 'Bt7' )
fig.set_size_inches( 8, 5.25 )
fig.clf()
ax = plt.gca()
ax.plot( x.t[0], x.b7[0], 
		label='Bt7 db' )
ax.set_title( shot )

fig = plt.figure( 'Bt8' )
fig.set_size_inches( 8, 5.25 )
fig.clf()
ax = plt.gca()
ax.plot( x.t[0], x.b8[0], 
		label='Bt8 db' )
ax.set_title( shot )

n_qsh = x.n_qsh[0]
t_qsh_begin = np.atleast_1d( x.t1_arr[0]*1E-3 )
t_qsh_end = np.atleast_1d( x.t2_arr[0]*1E-3 )

t_min = t_qsh_begin[0]
t_max = t_qsh_end[-1]

qshs = []
for i_qsh in range( n_qsh ) :
	qsh_name = 'T%02d' % i_qsh
	qshs.append( x[qsh_name][0] )

# per listare i nomi dei campi
print qshs[0].dtype.names

fig = plt.figure( 'Flux' )
fig.set_size_inches( 8, 7 )
fig.clf()
ax = plt.gca()
fig.subplots_adjust( top=0.95, bottom=0.08, left=0.08, right=0.95, hspace=0.2, wspace=0.2 )
ax.contour( qshs[0].xxg[0], qshs[0].yyg[0], qshs[0].mapro[0][7] )
ax.set_aspect('equal', adjustable='box')
# ax.axis( 'equal' )
ax.set_title( shot )

tree  = mds.Tree("RFX", shot )
alpha_tor = tree.getNode( r'\A::ALPHA_TOR' )
Theta_0_tor = tree.getNode( r'\A::THETA0_TOR' )

alpha = tree.getNode( r'\A::ALPHA' )
Theta_0 = tree.getNode( r'\A::THETA_R0' )

try:
	alpha_tor.data()
except:
	print "no tor eq data"

fig = plt.figure( 'AT0' )
fig.set_size_inches( 8, 5.25 )
fig.clf()
ax = plt.gca()

ax.plot( alpha.dim_of().data(), alpha.data(), label=r'$\alpha$ cyl' )
ax.plot( alpha_tor.dim_of().data(), alpha_tor.data(), label=r'$\alpha$ tor' )

ax.plot( Theta_0.dim_of().data(), Theta_0.data(), label=r'$\Theta_{0}$ cyl' )
ax.plot( Theta_0_tor.dim_of().data(), Theta_0_tor.data(), label=r'$\Theta_{0}$ tor' )

ax.set_title( shot )
ax.set_ylim( -0.2, 12. )
ax.set_xlim( -0.05, 0.55 )
ax.legend()

