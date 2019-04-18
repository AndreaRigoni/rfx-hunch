
import matplotlib.pyplot as plt
from scipy.io import readsav

import MDSplus as mds
import sys 

# funziona solo con A aggiornato!
# dare da linea di comando
# $ treeSetSource A inProgress prima di iniziare

# prende il numero di shot come parametro
# oppure se non presente prova a verificare se la variabile e' gia' definita
# per usare la variabile shot pre-esistente 
# dare da ipython %run -i spectrum_test

argv = sys.argv 

if len( argv ) > 1 :
	try:
		shot = int( argv[1] )
	except:
		print "invalid shot: ", argv[1]
		sys.exit(0)

try:
	shot
except:
	shot = 36239
	print 'shot default:'

if ( shot < 15600 or shot > 39391 ) :
	print "invalid shot num: ", shot
	sys.exit(0)

print shot

#shot = 36239

tree = mds.Tree("RFX", shot )

#m1 = Data.compile("ModesBt( 0, 0., 0.5, 0., 1, 0,-7,-7)")

#m1.plot()
#m0 = Data.compile("ModesBt( 1, 0., 0.5, 0., 0, 0,-7,-7)")

# m0.plot()
#m0 = mds.Data.execute("ModesBr( 0 ,0.01, 0.5, 0, 0, 0 , -7, -7 )")
#m1 = mds.Data.execute("ModesBr( 0 ,0.01, 0.5, 0, 1, 0 , -7, -7 )")

# spectrum(keep, t1, t2, dt, n, m, codice, OPZIONALE correction_code, OPZIONALE eq_nct, OPZIONALE n1, OPZIONALE n2)
# codice identifica la quantita' da restituire
# 0 -> raggio di risonanza
# +1 -> absbt alle misure
# -1 -> argbt alle misure (normalizzato a 180) + m
# +2 -> absbr alle misure
# -2 -> argbr alle misure (normalizzato a 180) + m
# +3 -> absbr alla risonanza
# -3 -> argbr alla risonanza (normalizzato a 180) + m
# +4 -> absflux alla risonanza
# -4 -> argflux alla risonanza (normalizzato a 180) + m
# +5 -> absbr al massimo
# +6 -> absflux al massimo
# +7 -> absbt alle misure in approssimazione cilindrica
# -7 -> argbt alle misure in approssimazione cilindrica (normalizzato a 180)
# +12 -> absbr a raggio plasma
# -12 -> argbr a raggio plasma (normalizzato a 180) + m

t_start = 0.1
t_end = 0.14
correction = 1

rs_1_7 =       mds.Data.execute("spectrum( 0 , %6.4f, %6.4f, .0005, -7, 1,  0, %d, 1 )" % (t_start, t_end, correction ) )
absBr_rs_1_7 = mds.Data.execute("spectrum( 1 , %6.4f, %6.4f, .0005, -7, 1,  3, %d, 1 )"  % (t_start, t_end, correction ) )
argBr_rs_1_7 = mds.Data.execute("spectrum( 1 , %6.4f, %6.4f, .0005, -7, 1, -3, %d, 1 )" % (t_start, t_end, correction ) )

absBt_rm_1_7 = mds.Data.execute("spectrum( 1 , %6.4f, %6.4f, .0005, -7, 1,  1, %d, 1 )" % (t_start, t_end, correction ) )
argBt_rm_1_7 = mds.Data.execute("spectrum( 1 , %6.4f, %6.4f, .0005, -7, 1, -1, %d, 1 )" % (t_start, t_end, correction ) )

absBr_rm_1_7 = mds.Data.execute("spectrum( 1 , %6.4f, %6.4f, .0005, -7, 1,  2, %d, 1 )" % (t_start, t_end, correction ) )
argBr_rm_1_7 = mds.Data.execute("spectrum( 1 , %6.4f, %6.4f, .0005, -7, 1, -2, %d, 1 )" % (t_start, t_end, correction ) )

absBt_rm_1_7_tc = mds.Data.execute("spectrum( 1 , %6.4f, %6.4f, .0005, -7, 1,  7, %d, 1 )" % (t_start, t_end, correction ) )
argBt_rm_1_7_tc = mds.Data.execute("spectrum( 1 , %6.4f, %6.4f, .0005, -7, 1, -7, %d, 1 )" % (t_start, t_end, correction ) )

absBt_rm_1_8    = mds.Data.execute("spectrum( 1 , %6.4f, %6.4f, .0005, -8, 1,  1, %d, 1 )" % (t_start, t_end, correction ) )
absBt_rm_1_8_tc = mds.Data.execute("spectrum( 1 , %6.4f, %6.4f, .0005, -8, 1,  7, %d, 1 )" % (t_start, t_end, correction ) )

absBt_rm_1_7_cyl = mds.Data.execute("spectrum_cyl( 0 , %6.4f, %6.4f, .0005, -7, 1,  1, %d )" % (t_start, t_end, correction ) )

fig = plt.figure( 'Bt7' )
ax = plt.gca()
ax.plot( absBt_rm_1_7.dim_of().data(), absBt_rm_1_7.data().ravel(),
		label='Bt7 tor %d' % correction )
ax.plot( absBt_rm_1_7_tc.dim_of().data(), absBt_rm_1_7_tc.data().ravel(), 
		label='Bt7 tc %d' % correction )
ax.plot( absBt_rm_1_7_cyl.dim_of().data(), absBt_rm_1_7_cyl.data().ravel(),
		label='Bt7 cyl %d' % correction )

ax.set_title( shot )
ax.legend()

# ---

fig = plt.figure( 'Bt8' )
ax = plt.gca()
ax.plot( absBt_rm_1_8.dim_of().data(), absBt_rm_1_8.data().ravel(), color='r',
		label='Bt8 tor' )
ax.plot( absBt_rm_1_8_tc.dim_of().data(), absBt_rm_1_8_tc.data().ravel(), color='g',
		label='Bt8 tc' )

ax.set_title( shot )
ax.legend()

