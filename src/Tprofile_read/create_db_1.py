import numpy as np
from scipy.io import readsav
import sys,os

from Hunch_utils import * 




def read_spectrum(shot, connection=None, server='rat2:52368', t0=10, t1=20, dt=5, mode_n=range(7,17), mode_m=1, correction=3):
	import MDSplus as mds
	tree = None
	if connection is None:
		cn = mds.Connection(server)
	else:
		cn = connection		
	tree = cn.openTree('RFX', shot)

	# convert to [ms]*E-1  decimi di millisecondo
	t0 = float(t0)*1E-4
	t1 = float(t1)*1E-4
	dt = float(dt)*1E-4

	m = mode_m
	class Spectrum(Struct):
		n           = [np.nan] * len(mode_n)
		m           = [np.nan] * len(mode_n)
		rs          = [np.nan] * len(mode_n)
		dimof       = [np.nan] * len(mode_n)
		absBt_rm    = [np.nan] * len(mode_n)
		argBt_rm    = [np.nan] * len(mode_n)
		absBr_rm    = [np.nan] * len(mode_n)
		argBr_rm    = [np.nan] * len(mode_n)
		absFlux_rs  = [np.nan] * len(mode_n)
		argFlux_rs  = [np.nan] * len(mode_n)
		absBr_rs    = [np.nan] * len(mode_n)
		argBr_rs    = [np.nan] * len(mode_n)
		absBr_rp    = [np.nan] * len(mode_n)
		argBr_rp    = [np.nan] * len(mode_n)
		absBr_max   = [np.nan] * len(mode_n)
		absFlux_max = [np.nan] * len(mode_n)

	data = Spectrum()
	
	n1 = min(mode_n)
	n2 = max(mode_n)
	# precalculate code variables for all toroidal modes from n1 to n2
	# print("spectrum( 0 , %6.4f, %6.4f, %6.4f, %d, %d,   0, %d, 1, %d, %d )" % (t0, t1, dt, -7, 1, correction, n1, n2) )
	#  0 -> raggio di risonanza
	# +1 -> absbt alle misure
	# -1 -> argbt alle misure (normalizzato a 180°) + m
	# +2 -> absbr alle misure
	# -2 -> argbr alle misure (normalizzato a 180°) + m
	# +3 -> absbr alla risonanza
	# -3 -> argbr alla risonanza (normalizzato a 180°) + m
	# +4 -> absflux alla risonanza
	# -4 -> argflux alla risonanza (normalizzato a 180°) + m
	# +5 -> absbr al massimo
	# +6 -> absflux al massimo
	# +7 -> absbt alle misure in approssimazione cilindrica
	# -7 -> argbt alle misure in approssimazione cilindrica (normalizzato a 180°)
	# +12 -> absbr a raggio plasma
	# -12 -> argbr a raggio plasma (normalizzato a 180°) + m
	cn.get("spectrum( 0 , %6.4f, %6.4f, %6.4f, %d, %d,   3, %d, 1, %d, %d )" % (t0, t1, dt, -7, 1, correction, n1, n2 ) )
	for i,n in enumerate(mode_n):
		data.n          [i] = int(n)
		data.m          [i] = int(m)
		data.rs         [i] = cn.get("_a = spectrum( 1 , %6.4f, %6.4f, %6.4f, %d, %d,  0, %d, 1); _a" % (t0, t1, dt, -n, m, correction ) ).ravel()
		data.dimof      [i] = np.rint(cn.get("dim_of(_a)")	* 1E4)
		data.absBt_rm   [i] = cn.get("spectrum( 1 , %6.4f, %6.4f, %6.4f, %d, %d,   1, %d, 1 )" % (t0, t1, dt, -n, m, correction ) ).ravel()
		data.argBt_rm   [i] = cn.get("spectrum( 1 , %6.4f, %6.4f, %6.4f, %d, %d,  -1, %d, 1 )" % (t0, t1, dt, -n, m, correction ) ).ravel()
		data.absBr_rm   [i] = cn.get("spectrum( 1 , %6.4f, %6.4f, %6.4f, %d, %d,   2, %d, 1 )" % (t0, t1, dt, -n, m, correction ) ).ravel()
		data.argBr_rm   [i] = cn.get("spectrum( 1 , %6.4f, %6.4f, %6.4f, %d, %d,  -2, %d, 1 )" % (t0, t1, dt, -n, m, correction ) ).ravel()
		data.absBr_rs   [i] = cn.get("spectrum( 1 , %6.4f, %6.4f, %6.4f, %d, %d,   3, %d, 1 )" % (t0, t1, dt, -n, m, correction ) ).ravel()
		data.argBr_rs   [i] = cn.get("spectrum( 1 , %6.4f, %6.4f, %6.4f, %d, %d,  -3, %d, 1 )" % (t0, t1, dt, -n, m, correction ) ).ravel()
		data.absFlux_rs [i] = cn.get("spectrum( 1 , %6.4f, %6.4f, %6.4f, %d, %d,   4, %d, 1 )" % (t0, t1, dt, -n, m, correction ) ).ravel()
		data.argFlux_rs [i] = cn.get("spectrum( 1 , %6.4f, %6.4f, %6.4f, %d, %d,  -4, %d, 1 )" % (t0, t1, dt, -n, m, correction ) ).ravel()
		data.absBr_rp   [i] = cn.get("spectrum( 1 , %6.4f, %6.4f, %6.4f, %d, %d,  12, %d, 1 )" % (t0, t1, dt, -n, m, correction ) ).ravel()
		data.argBr_rp   [i] = cn.get("spectrum( 1 , %6.4f, %6.4f, %6.4f, %d, %d, -12, %d, 1 )" % (t0, t1, dt, -n, m, correction ) ).ravel()
		data.absBr_max  [i] = cn.get("spectrum( 1 , %6.4f, %6.4f, %6.4f, %d, %d,   5, %d, 1 )" % (t0, t1, dt, -n, m, correction ) ).ravel()
		data.absFlux_max[i] = cn.get("spectrum( 1 , %6.4f, %6.4f, %6.4f, %d, %d,   6, %d, 1 )" % (t0, t1, dt, -n, m, correction ) ).ravel()


	if tree is not None:
		tree.close()
	return data





def read_te_prof( shot, data_dir='/scratch/gobbin/rigoni/', add_spectrum=False ) :
	file = 'dsx3_%d.sav' % shot
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

	t_min  = t_qsh_begin[0]
	t_max  = t_qsh_end[-1]
	
	dim_pulse = x.t[0]
	def find_pos(t): 
		return np.argmin(dim_pulse - t)
	
	qshs = []
	for i_qsh in range( n_qsh ) :
		qsh_name = 'T%02d' % i_qsh
		qshs.append( x[qsh_name][0] )

	# per listare i nomi dei campi
	# print qshs[0].dtype.names
	# per la label faccio shot_tttt 

	# (fieldname, datatype, shape)
	sample_dtype = np.dtype( [ ('label','S10'),
							('pulse', np.int32),
							('start', np.int32),
							('i_qsh', np.int32 ), 
							('tbordo','>f4' ),
							('tcentro','>f4' ),
							('pos','>f4' ),
							('grad','>f4' ),
							('n_ok', np.int32 ),
							('prel','>f4', (20,) ),
							('rho','>f4', (20,) ),
							('te','>f4', (20,) ),
							
							('Ip','>f4' ),
							('dens', '>f4'),
							('Te_dsxm', '>f4'),
							('F',   '>f4'),
							('TH',  '>f4'),
							('POW', '>f4'),
							('VT',  '>f4'),
							('VP',  '>f4'),

							('B0',  '>f4'),
							('B07', '>f4'),
							('B08', '>f4'),
							('B06', '>f4'),
							('B1',  '>f4'),
							('B17', '>f4'),
							('B18', '>f4'),
							('B19', '>f4'),
							('NS',  '>f4'),


							# SHEq map
							('mapro','>f4', (51,51) ),
							('xxg','>f4', (51,) ),
							('yyg','>f4', (51,) ),

							# Spectrum
							('n', '>i4', (10,) ),
							('absBt_rm', '>f4', (10,)),
							('argBt_rm', '>f4', (10,)),
							('absBr_rm', '>f4', (10,)),
							('argBr_rm', '>f4', (10,)),
							('absFlux_rs', '>f4', (10,)),
							('argFlux_rs', '>f4', (10,)),
							('absBr_rs', '>f4', (10,)),
							('argBr_rs', '>f4', (10,)),
							('absBr_rp', '>f4', (10,)),
							('argBr_rp', '>f4', (10,)),
							('absBr_max', '>f4', (10,)),
							('absFlux_max', '>f4', (10,)),
						] )

	n_times_tot = 0
	for qsh in qshs :
		n_times_tot += qsh.tempi[0].shape[0]

	q_data = np.empty( n_times_tot, dtype=sample_dtype )

	i_time = 0
	i_qsh  = 0
	for qsh in qshs : 
		n_times = qsh.tempi[0].shape[0]
		t0 = np.rint(np.min(qsh.tempi[0])*1E4)
		t1 = np.rint(np.max(qsh.tempi[0])*1E4)
		if add_spectrum:
			spectrum = read_spectrum(shot, t0=t0, t1=t1)

		for k_time in np.arange( n_times ) :
			tx = qsh.tempi[0][k_time]			
			tt = np.rint( tx*1E4 ) # label tempo in decimi di ms
			ii = find_pos(tx)      # posizione nell'array della scarica

			label = r'%5d_%04d' % ( shot, tt )
			#print label
			
			q_data[i_time]['label'] = label
			q_data[i_time]['pulse'] = shot
			q_data[i_time]['start'] = tt
			q_data[i_time]['i_qsh'] = i_qsh
			
			te_ok = qsh.te3[0][k_time,:] > 0
			
			q_data[i_time]['n_ok'] = np.count_nonzero( te_ok )
			q_data[i_time]['prel'] = qsh.prel3[0][k_time]
			q_data[i_time]['rho']  = qsh.rho3[0][k_time]
			q_data[i_time]['te']   = qsh.te3[0][k_time]
			
			q_data[i_time]['Ip']      = x.ip[0][ii]
			q_data[i_time]['dens']    = x.dens[0][ii]
			q_data[i_time]['Te_dsxm'] = x.te[0][ii]
			q_data[i_time]['F']    = x.f[0][ii]
			q_data[i_time]['TH']    = x.th[0][ii]
			q_data[i_time]['POW']  = x.POW[0][ii]
			q_data[i_time]['VT']   = x.vt[0][ii]
			q_data[i_time]['VP']   = x.vp[0][ii]

			q_data[i_time]['B0']   = x.b0[0][ii]
			q_data[i_time]['B07']  = x.b07[0][ii]
			q_data[i_time]['B08']  = x.b08[0][ii]
			q_data[i_time]['B06']  = x.b06[0][ii]
			q_data[i_time]['B1']   = x.bs[0][ii]
			q_data[i_time]['B17']  = x.b7[0][ii]
			q_data[i_time]['B18']  = x.b8[0][ii]
			q_data[i_time]['B19']  = x.b9[0][ii]
			q_data[i_time]['NS']   = x.ns[0][ii]

			q_data[i_time]['tbordo'] = qsh.tbordo[0][k_time]
			q_data[i_time]['tcentro'] = qsh.tcentro[0][k_time]
			q_data[i_time]['pos'] = qsh.pos2[0][k_time]
			q_data[i_time]['grad'] = qsh.grad2[0][k_time]
						
			# SHEq
			q_data[i_time]['mapro'] = qsh.mapro[0][k_time]
			q_data[i_time]['xxg'] = qsh.xxg[0]
			q_data[i_time]['yyg'] = qsh.yyg[0]			

			# spectrum
			q_data[i_time]['n'] = 0 
			q_data[i_time]['absBt_rm'][:] = np.nan
			q_data[i_time]['argBt_rm'][:] = np.nan
			q_data[i_time]['absBr_rm'][:] = np.nan
			q_data[i_time]['argBr_rm'][:] = np.nan
			q_data[i_time]['absFlux_rs'][:] = np.nan
			q_data[i_time]['argFlux_rs'][:] = np.nan
			q_data[i_time]['absBr_rs'][:] = np.nan
			q_data[i_time]['argBr_rs'][:] = np.nan
			q_data[i_time]['absBr_rp'][:] = np.nan
			q_data[i_time]['argBr_rp'][:] = np.nan
			q_data[i_time]['absBr_max'][:] = np.nan
			q_data[i_time]['absFlux_max'][:] = np.nan

			if add_spectrum:
				spid = list(spectrum.dimof[0]).index(tt)
				q_data[i_time]['n'][:len(spectrum.n)] = spectrum.n
				q_data[i_time]['absBt_rm'][:len(spectrum.n)] = np.transpose(spectrum.absBt_rm)[spid]
				q_data[i_time]['argBt_rm'][:len(spectrum.n)] = np.transpose(spectrum.argBt_rm)[spid]
				q_data[i_time]['absBr_rm'][:len(spectrum.n)] = np.transpose(spectrum.absBr_rm)[spid]
				q_data[i_time]['argBr_rm'][:len(spectrum.n)] = np.transpose(spectrum.argBr_rm)[spid]
				q_data[i_time]['absFlux_rs'][:len(spectrum.n)] = np.transpose(spectrum.absFlux_rs)[spid]
				q_data[i_time]['argFlux_rs'][:len(spectrum.n)] = np.transpose(spectrum.argFlux_rs)[spid]
				q_data[i_time]['absBr_rs'][:len(spectrum.n)] = np.transpose(spectrum.absBr_rs)[spid]
				q_data[i_time]['argBr_rs'][:len(spectrum.n)] = np.transpose(spectrum.argBr_rs)[spid]
				q_data[i_time]['absBr_rp'][:len(spectrum.n)] = np.transpose(spectrum.absBr_rp)[spid]
				q_data[i_time]['argBr_rp'][:len(spectrum.n)] = np.transpose(spectrum.argBr_rp)[spid]
				q_data[i_time]['absBr_max'][:len(spectrum.n)] = np.transpose(spectrum.absBr_max)[spid]
				q_data[i_time]['absFlux_max'][:len(spectrum.n)] = np.transpose(spectrum.absFlux_max)[spid]
			
			i_time += 1
		
		i_qsh += 1

	return q_data

# ------------------------------------------------------------------------------

fail_shots = [29961]

def create_db(add_spectrum=False):
	from glob import glob

	abs_srcdir = os.environ.get('abs_top_srcdir')
	data_dir = abs_srcdir + '/data/gobbin_db/'
	file_list = glob( data_dir+r'dsx3*')

	q_list = []
	n_shots = 0	
	for fname in file_list[78:]:
		shot_char_pos = fname.find( r'dsx3_' )+5
		shot = np.int( fname[shot_char_pos:shot_char_pos+5] )
		if shot not in fail_shots:
			q_list.append( read_te_prof( shot, data_dir, add_spectrum ) )
			n_shots +=1

	q_all_data = np.concatenate( q_list )

	np.save( 'te_db_1', q_all_data, allow_pickle=False )
	print ('Saved %d profiles from %d shots.' % ( q_all_data.shape[0], n_shots ))


# if __name__ == '__main__':
# 	create_db()

