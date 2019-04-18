# prova di implementazione algoritmo RANSAC su un caso di esempio

from __future__ import division, print_function

import matplotlib.pyplot as plt
from scipy.io import readsav
import numpy as np

plt.ion()

#import sys 
#import MDSplus as mds


rho = np.array( [ 0.64007366,  0.55654508,  0.37148875,  0.28707314,  0.23085016,
        0.19731766,  0.17019749,  0.14400813,  0.11366016,  0.07045307,
        0.02995016,  0.1533501 ,  0.27953583,  0.39990026,  0.50619876,
        0.60163134], dtype=np.float32)

te = np.array([ 504.97167969,  303.62905884,  529.33563232,  538.14001465,
        537.16394043,  524.76824951,  568.06292725,  646.22613525,
        750.49383545,  944.4630127 ,  928.89984131,  723.57531738,
        608.80743408,  529.07507324,  572.19647217,  405.27713013], dtype=np.float32)

fig = plt.figure( 'rho' )
#fig.set_size_inches( 11, 4 )
fig.set_size_inches( 5, 4 )
fig.clf()
#ax1 = plt.subplot( 121 )
ax1 = plt.gca()
ax1.plot( rho, 'o' )
ax1.set_ylabel( r'$\rho$' )
# ax.set_title( shot )

k_ext = rho > 0.4

te_ext_ini = te[k_ext].mean()
delta_te_ini =  te.max() - te_ext_ini

rho_0_ini = 0.2
grad_te_ini = -5000.

fig = plt.figure( 'RANSAC_1' )
fig.set_size_inches( 5, 4 )
fig.clf()
# ax2 = plt.subplot( 122 )
ax2 = plt.gca()

ax2.plot( rho, te, '.' )
ax2.set_ylabel( r'Te' )
ax2.set_xlabel( r'$\rho$' )

alpha_ini = grad_te_ini / delta_te_ini
xp = np.linspace( 0., np.ceil( rho.max()*10. )/10. )
yp = te_ext_ini + 0.5*delta_te_ini * ( 1 + np.tanh( alpha_ini*(xp - rho_0_ini)  ) )
ax2.plot( xp, yp )

from scipy.optimize import curve_fit

def fit_func( rho_f, rho_0, te_ext, delta_te, grad_te ) :
	alpha = grad_te / delta_te
	te = te_ext + 0.5*delta_te * ( 1 + np.tanh( alpha*(rho_f - rho_0) ) )
	return te

p0 = ( rho_0_ini, te_ext_ini, delta_te_ini, grad_te_ini )
pfit, pcov = curve_fit( fit_func, rho, te, p0=p0 )

ax2.plot( xp, fit_func( xp, *pfit ), 'b' )

# stop
# funzione di voto del ransac
# def eval_func( rho_f, te_f, inlier_threshold, rho_0, te_ext, delta_te, grad_te ) :
	#alpha = grad_te / delta_te
	#delta_te = te_f - ( te_ext + 0.5*delta_te * ( 1 + np.tanh( alpha*(rho_f - rho_0) ) ) )
	#inliers = np.abs( delta_te ) <  inlier_threshold
	#return np.count_nonzero( inliers )

def ransac_te( rho_f, te_f, fit_fn, p0, min_inliers=12, samples_to_fit=10,
		   inlier_threshold=50., max_iters=100 ) :    
	
	best_pfit = None
	best_model_performance = 0
	
	num_samples = rho_f.shape[0]
	
	for i in range(max_iters):
		# model fit
		sample = np.random.choice( num_samples, size=samples_to_fit, replace=False )
		try:
			pfit, pcov = curve_fit( fit_func, rho_f[sample], te_f[sample], p0=p0 )
		except :
			print( i, "sample fit failed" )
			continue
		
		# model performance
		# model_performance = evaluate_fn( rho_f, te_f, inlier_threshold, p_fit )
		delta_te = te_f - fit_func( rho_f, *pfit )
		inliers = np.abs( delta_te ) <  inlier_threshold
		model_performance = np.count_nonzero( inliers )
		
		print( i, model_performance )
		if model_performance < min_inliers :
			continue
		
		if model_performance > best_model_performance :
			best_pfit = pfit
			best_inliers = inliers
			best_sample = sample
			best_model_performance = model_performance
	
	return best_pfit, best_inliers, best_sample

bfit, inliers, sample = ransac_te( rho, te, fit_func, p0, min_inliers=12, samples_to_fit=10,
		   inlier_threshold=50., max_iters=100 )

ax2.plot( xp, fit_func( xp, *bfit ), 'r' )
ax2.plot( rho[inliers], te[inliers], 'x' )