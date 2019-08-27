import numpy as np
from scipy.io import readsav
import sys,os

from Hunch_utils import * 

fail_shots = [34820, 30861]


def read_te_prof(shot, data_dir):
    file = 'thomson_%d.sav' % shot
    try: x = readsav( data_dir+file, python_dict=False )
    except:
        print('file not found: ', file)
        sys.exit(0)
    print(shot)
    
    def pad(d, size=200):
        c = np.empty(size)
        c.fill(np.nan)
        c[ :d.shape[0] ] = d
        return c

    # find_pos = lambda t: np.argmin(x.t - t)
    sample_dtype = np.dtype([('label','S10'),
							 ('pulse', np.int32),
							    ('start', np.int32),
                                ('t',     np.float32 ),
                                ('r','>f4', (200,) ),
                                ('te_r','>f4', (200,) ),
                            ])

    q_data = np.empty( len(x.t), dtype=sample_dtype)
    for i,t in enumerate(x.t):
        tstart = np.rint( t*1E4 )
        q_data[i]['label'] = r'%5d_%04d' % ( shot, tstart )
        q_data[i]['pulse'] = shot
        q_data[i]['start'] = tstart
        q_data[i]['t']     = t
        q_data[i]['r']     = pad(x.r)
        q_data[i]['te_r']  = pad(x.te_r[i])
    return q_data


def create_db():
    from glob import glob
    import ntpath
    abs_srcdir = os.environ.get('abs_top_srcdir')
    data_dir = abs_srcdir + '/data/thomson/'
    file_list = glob( data_dir+r'thomson_*')
    q_list  = []
    n_shots = 0	

    for fname in file_list[:]:
            shot_char_pos = fname.find( r'thomson_' )+len('thomson_')
            shot = np.int( fname[shot_char_pos:shot_char_pos+5] )
            if shot not in fail_shots:
                q_list.append( read_te_prof( shot, data_dir ) )
            n_shots += 1
    q_all_data = np.concatenate( q_list )
    np.save( 'ts_db_1', q_all_data, allow_pickle=False )
    print ('Saved %d profiles from %d shots.' % ( q_all_data.shape[0], n_shots ))



