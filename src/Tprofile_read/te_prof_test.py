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

# almeno verifico che sia un impulso di RFX-mod	
if ( shot < 15600 or shot > 39391 ) :
	print "invalid shot num: ", shot
	sys.exit(0)

file = 'dsx3_%d.sav' % shot
print file

try:
	x = readsav( data_dir+file, python_dict=False ).st
	# x = readsav( data_dir+file, python_dict=True )
except:
	print "file not found: ", file
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

# per la label faccio shot_tttt 

i_qsh = 0
#i_time = 8
i_time = 36
tf = qshs[i_qsh].tempi[0][i_time]

# label tempo in decimi di ms
tttt = np.int( tf*1E4 )

case_label = r'%5d_%04d' % ( shot, tttt )
# contour della topologia della mappa di flusso
fig = plt.figure( 'Flux' )
fig.set_size_inches( 6, 5 )
fig.clf()
ax = plt.gca()
fig.subplots_adjust( top=0.95, bottom=0.08, left=0.08, right=0.95, hspace=0.2, wspace=0.2 )
ax.contour( qshs[i_qsh].xxg[0], qshs[i_qsh].yyg[0], qshs[i_qsh].mapro[0][i_time] )
ax.set_aspect('equal', adjustable='box')
# ax.axis( 'equal' )
ax.set_title( case_label )


# Esempio plot temperatura DSX3:
# IDL> plot,st.t00.prel3[*,10],st.t00.te3[*,10],ps=4
fig = plt.figure( 'Te' )
fig.set_size_inches( 9, 4 )
fig.clf()
fig.suptitle( case_label )
fig.subplots_adjust( top=0.92, bottom=0.12, left=0.08, right=0.97, 
	hspace=0.2, wspace=0.15 )

te_ok = qshs[i_qsh].te3[0][i_time,:] > 0

ax1 = plt.subplot(121)
#ax1.plot( qshs[i_qsh].prel3[0][i_time,1:], qshs[i_qsh].te3[0][i_time,1:], "o" )
ax1.plot( qshs[i_qsh].prel3[0][i_time,te_ok], qshs[i_qsh].te3[0][i_time,te_ok], "o" )
#ax.plot( qshs[i_qsh].prel3[0][i_time,:], qshs[i_qsh].te3[0][i_time,:], marker="o", linestyle='None' )
ax1.set_xlabel( r'impact parameter' )
ax1.set_ylabel( r'Te SXR [eV]' )


# stesso profilo rimappato:
# IDL plot,st.t00.rho3[*,10],st.t00.te3[*,10],ps=4,/yst
ax2 = plt.subplot(122, sharey=ax1 )
ax2.plot( qshs[i_qsh].rho3[0][i_time,te_ok], qshs[i_qsh].te3[0][i_time,te_ok], "o" )
ax2.set_xlabel( r'$\rho$' )
ax2.set_ylim(0)

print qshs[i_qsh].tcentro[0][i_time]
print qshs[i_qsh].tbordo[0][i_time]
print qshs[i_qsh].grad2[0][i_time]
print qshs[i_qsh].pos2[0][i_time]
