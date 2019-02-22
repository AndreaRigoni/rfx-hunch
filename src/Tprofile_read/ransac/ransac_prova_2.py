# prova algoritmo RANSAC su un po' di dati
from __future__ import division, print_function

import matplotlib.pyplot as plt
from scipy.io import readsav
import numpy as np

plt.ion()

import sys 
#import MDSplus as mds

from scipy.optimize import curve_fit

def fit_func( rho_f, rho_0, te_ext, delta_te, grad_te ) :
	alpha = grad_te / delta_te
	te = te_ext + 0.5*delta_te * ( 1 + np.tanh( alpha*(rho_f - rho_0) ) )
	return te

# funzione di voto del ransac
#def eval_func( rho_f, te_f, inlier_threshold, rho_0, te_ext, delta_te, grad_te ) :
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
		
		# print( i, model_performance )
		if model_performance < min_inliers :
			continue
		
		if model_performance > best_model_performance :
			best_pfit = pfit
			best_inliers = inliers
			best_sample = sample
			best_model_performance = model_performance
	
	return best_pfit, best_inliers, best_sample

# -----------------------------------------------------------------------------

data_dir = '/scratch/gobbin/rigoni/'


shot_list = [ 30808 ]

fig = plt.figure( 'RANSAC_2' )
fig.set_size_inches( 6, 4.5 )

for shot in shot_list :
	print( shot )
	
	if ( shot < 15600 or shot > 39391 ) :
		print( "invalid shot num: ", shot )
		continue
	
	file = 'dsx3_%d.sav' % shot
	print( file )
	try:
		x = readsav( data_dir+file, python_dict=False ).st
		# x = readsav( data_dir+file, python_dict=True )
	except:
		print( "invalid file: ", file )
		sys.exit(0)
	
	n_qsh = x.n_qsh[0]
	t_qsh_begin = np.atleast_1d( x.t1_arr[0]*1E-3 )
	t_qsh_end = np.atleast_1d( x.t2_arr[0]*1E-3 )

	t_min = t_qsh_begin[0]
	t_max = t_qsh_end[-1]

	qshs = []
	for i_qsh in range( n_qsh ) :
		qsh_name = 'T%02d' % i_qsh
		qshs.append( x[qsh_name][0] )

	i_qsh = 0
	n_times = qshs[i_qsh].tempi[0].shape[0] 
	print( n_times )
	
	for i_time in np.arange( n_times ) :
		tf = qshs[i_qsh].tempi[0][i_time]

		# label tempo in decimi di ms
		tttt = np.int( tf*1E4 )

		case_label = r'%5d_%04d' % ( shot, tttt )

		te_ok = qshs[i_qsh].te3[0][i_time,:] > 0

		te = qshs[i_qsh].te3[0][i_time,te_ok]
		rho = qshs[i_qsh].rho3[0][i_time,te_ok]
		# rb = qshs[i_qsh].prel3[0][i_time,te_ok]

		print( qshs[i_qsh].tcentro[0][i_time] )
		print( qshs[i_qsh].tbordo[0][i_time] )
		print( qshs[i_qsh].grad2[0][i_time] )
		print( qshs[i_qsh].pos2[0][i_time] )

		# -----------------------------------------------------------------------------
		
		k_ext = rho > 0.4

		te_ext_ini = te[k_ext].mean()
		delta_te_ini =  te.max() - te_ext_ini

		rho_0_ini = 0.2
		grad_te_ini = -5000.
		fig = plt.figure( 'RANSAC_2' )
		fig.clf()
		# ax2 = plt.subplot( 122 )
		ax2 = plt.gca()

		ax2.plot( rho, te, '.' )
		ax2.set_ylabel( r'Te' )
		ax2.set_xlabel( r'$\rho$' )
		ax2.set_title( case_label )
		
		alpha_ini = grad_te_ini / delta_te_ini
		xp = np.linspace( 0., np.ceil( rho.max()*10. )/10. )
		yp = te_ext_ini + 0.5*delta_te_ini * ( 1 + np.tanh( alpha_ini*(xp - rho_0_ini)  ) )
		ax2.plot( xp, yp )

		p0 = ( rho_0_ini, te_ext_ini, delta_te_ini, grad_te_ini )
		try:
			pfit, pcov = curve_fit( fit_func, rho, te, p0=p0 )
			ax2.plot( xp, fit_func( xp, *pfit ), 'b' )
		except :
			print( case_label, " VERY BAD!" )
			plt.pause(0.8)
			continue
		
		try:
			bfit, inliers, sample = ransac_te( rho, te, fit_func, p0, min_inliers=10, samples_to_fit=10,
				inlier_threshold=50., max_iters=100 )

			ax2.plot( xp, fit_func( xp, *bfit ), 'r' )
			ax2.plot( rho[inliers], te[inliers], 'x' )
		except:
			print( case_label, " VERY BAD!" )
			plt.pause(0.8)
			
		plt.pause(0.8)

		
		
