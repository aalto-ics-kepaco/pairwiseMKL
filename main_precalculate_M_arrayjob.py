from sys import argv, exit
import os
import numpy as np
from pairwisemkl.learner.compute_M__arrayjob import *


try:
    id_in = int(argv[1])
except:
    exit()
    
    
data_path = './drug_response_data'


# Drug kernels
# Read file names of drug kernels
fn_kd = open(data_path + '/Drug_kernels/Drug_kernel_file_names.txt', 'r')
kd_file_names = fn_kd.readlines()
fn_kd.close()
kd_file_names = [x.split('\n')[0] for x in kd_file_names]
# Prepare a list of drug kernels
kd_list = []
for kd in kd_file_names:
    f_kd = open(data_path + '/Drug_kernels/' + kd, 'r')
    kd_list.append(np.loadtxt(f_kd))
    f_kd.close()


# Cell line kernels
# Read file names of cell line kernels
fn_kc = open(data_path + '/Cell_line_kernels/Cell_kernel_file_names.txt', 'r')
kc_file_names = fn_kc.readlines() 
fn_kc.close()
kc_file_names = [x.split('\n')[0] for x in kc_file_names]
kc_list = []
# Prepare a list of cell line kernels
for kc in kc_file_names:
    f_kc = open(data_path + '/Cell_line_kernels/' + kc, 'r')
    kc_list.append(np.loadtxt(f_kc))
    f_kc.close()


# Compute a single row of the matrix M (indexed by an integer id_in) 
# Matrix M is needed to optimize pairwise kernel weights
m = compute_M_row(kd_list, kc_list, id_in)

new_path = data_path + "/M"
if not os.path.exists(new_path):
    os.makedirs(new_path)
np.savetxt(new_path + '/M__row_'+str(id_in)+'.txt', m, delimiter='\t')


print('\nSuccess!') 