import numpy as np
from scipy.io import readsav
import sys,os



def read_te_prof( shot, data_dir='/scratch/gobbin/rigoni/' ) :
	file = 'dsx3_%d.sav' % shot
	# print (file)
	#shot = 30008
	#file = r'shot_dsx3_%d.sav' % shot

	try:
		x = readsav( data_dir+file, python_dict=False ).st
		# x = readsav( data_dir+file, python_dict=True )
	except:
		print ("file not found: ", file)
		sys.exit(0)


	n_qsh = x.n_qsh[0]
	t_qsh_begin = np.atleast_1d( x.t1_arr[0]*1E-3 )
	t_qsh_end = np.atleast_1d( x.t2_arr[0]*1E-3 )
	print (file,' ',n_qsh)

	t_min = t_qsh_begin[0]
	t_max = t_qsh_end[-1]

	qshs = []
	for i_qsh in range( n_qsh ) :
		qsh_name = 'T%02d' % i_qsh
		qshs.append( x[qsh_name][0] )

	# per listare i nomi dei campi
	# print qshs[0].dtype.names
	# per la label faccio shot_tttt 

	# (fieldname, datatype, shape)
	sample_dtype = np.dtype( [ ('label','S10'),
							('i_qsh', np.int32 ), 
							('tbordo','>f4' ),
							('tcentro','>f4' ),
							('pos','>f4' ),
							('grad','>f4' ),
							('n_ok', np.int32 ),
							('prel','>f4', (20,) ),
							('rho','>f4', (20,) ),
							('te','>f4', (20,) ),
						] )

	n_times_tot = 0
	for qsh in qshs :
		n_times_tot += qsh.tempi[0].shape[0]

	q_data = np.empty( n_times_tot, dtype=sample_dtype )

	i_time = 0
	i_qsh = 0
	for qsh in qshs : 
		n_times = qsh.tempi[0].shape[0]

		for k_time in np.arange( n_times ) :
			tx = qsh.tempi[0][k_time]

			# label tempo in decimi di ms
			tttt = np.rint( tx*1E4 )

			label = r'%5d_%04d' % ( shot, tttt )
			#print label
			
			q_data[i_time]['label'] = label
			q_data[i_time]['i_qsh'] = i_qsh
			
			te_ok = qsh.te3[0][k_time,:] > 0
			
			q_data[i_time]['n_ok'] = np.count_nonzero( te_ok )
			q_data[i_time]['prel'] = qsh.prel3[0][k_time]
			q_data[i_time]['rho']  = qsh.rho3[0][k_time]
			q_data[i_time]['te']   = qsh.te3[0][k_time]
			
			q_data[i_time]['tbordo'] = qsh.tbordo[0][k_time]
			q_data[i_time]['tcentro'] = qsh.tcentro[0][k_time]
			q_data[i_time]['pos'] = qsh.pos2[0][k_time]
			q_data[i_time]['grad'] = qsh.grad2[0][k_time]

			i_time += 1
		
		i_qsh += 1

	return q_data

# ------------------------------------------------------------------------------

from glob import glob

abs_srcdir = os.environ.get('abs_top_srcdir')
data_dir = abs_srcdir + '/data/gobbin_db/'

file_list = glob( data_dir+r'dsx3*')

q_list = []
n_shots = 0
for fname in file_list :
	shot_char_pos = fname.find( r'dsx3_' )+5
	shot = np.int( fname[shot_char_pos:shot_char_pos+5] )
	# print shot
	q_list.append( read_te_prof( shot, data_dir ) )
	n_shots +=1

q_all_data = np.concatenate( q_list )

np.save( 'te_db_1', q_all_data, allow_pickle=False )
print ('Saved %d profiles from %d shots.' % ( q_all_data.shape[0], n_shots ))
