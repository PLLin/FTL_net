import os
from pathlib import Path
from data.data_pipe import load_bin, load_mx_rec

if __name__ == '__main__':
   
    rec_path = 'data/faces_emore'
    load_mx_rec(rec_path)
    bin_files = ['agedb_30', 'cfp_fp', 'lfw']
    for i in range(len(bin_files)):
        load_bin(os.path.join(rec_path, bin_files[i]+'.bin'), os.path.join(rec_path, bin_files[i]))
