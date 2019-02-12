from sys import argv, exit
import os
import numpy as np
import copy
from math import sqrt
from sklearn import preprocessing, metrics
from pairwisemkl.learner.compute_a_regression import *
from pairwisemkl.learner.optimize_kernel_weights import *
from pairwisemkl.learner.cg_kron_rls import CGKronRLS


try:
    i_out = int(argv[1]) 
except:
    exit()
    

data_path = './drug_response_data'

new_path = data_path + "/ArrayJob_results"
if i_out==0 and not os.path.exists(new_path):
    os.makedirs(new_path)


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


# Number of pairwise kernels 
P = len(kd_list)*len(kc_list)
# Generate pairwise kernel ids
kd_ids, kc_ids = np.unravel_index(np.arange(P), (len(kd_list),len(kc_list)), order = 'C')


# Assemble matrix M precomputed using array jobs
M = np.empty([P,P]); M[:] = np.NAN
for row in range(P):
    f_m = open(data_path + '/M/M__row_' + str(row) + '.txt')
    m = np.loadtxt(f_m)
    f_m.close()
    M[row, row:P] = m[row:P]
    M[row:P, row] = m[row:P]



# Labels
# Read matrix with drug responses in cancer cell lines
f = open(data_path + '/Labels.txt', 'r')  # rows: drugs, columns: cell lines
Y = np.loadtxt(f)
f.close()
# Number of drugs and cell lines
n_d, n_c = Y.shape
# Labels in the vector form
y_vec = Y.ravel(order = 'C')
# Create indicies
ids = np.arange(n_d*n_c)
drug_ids, cell_ids = np.unravel_index(ids, (n_d,n_c), order = 'C') 

# Remove missing values (if any) from the label vector as well as the corresponding ids
ids_known = ~np.isnan(y_vec)               
y_vec_known = y_vec[ids_known]
drug_ids_known = drug_ids[ids_known]
cell_ids_known = cell_ids[ids_known]


# Values for the regularization parameter \lambda 
# (to be optimized in the nested CV)
regparam = [10.**x for x in range(-5, 1)]



# CV (10 outer folds, 3 inner folds)
# Each outer CV loop is run as a separate array job

# Vector where predicted drug responses will be stored
y_pred_outer_vec = np.empty([len(ids[ids_known])]); y_pred_outer_vec[:] = np.NAN

# Read pre-defined outer folds used in the experiments presented in the pairwiseMKL paper
outer_folds = np.loadtxt(data_path + '/Folds/outer_folds.txt').astype(int)

# Outer CV loop
print('Outer loop ' + str(i_out+1) + '\n')

test_ids  = np.array(np.where(outer_folds==i_out)).squeeze()
train_ids = np.array(np.where(outer_folds!=i_out)).squeeze()
    
# Test data 
y_test        = y_vec_known[test_ids]  
drug_ids_test = drug_ids_known[test_ids]
cell_ids_test = cell_ids_known[test_ids]

# Training data
# Training labels in the vector form 
y_train = y_vec_known[train_ids]
# Training labels in the matrix form
Y_train = copy.deepcopy(Y)   
Y_train[drug_ids_test,cell_ids_test] = np.nan
drug_ids_train = drug_ids_known[train_ids]
cell_ids_train = cell_ids_known[train_ids]

        
# Stage 1 of determining pairwise kernel weights 

# Compute vector a needed to optimize kernel weights
a = compute_a_regression(kd_list, kc_list, Y_train)

# Optimize pairwise kernel weights
k_weights_outer = optimize_kernel_weights(a, M)
    

# Stage 2 of pairwise model training 
    
# Find pairwise kernels with weights different from 0 (>10**-3).
ix = np.where(k_weights_outer[0,:] > 10**-3)[0]
# Corresponding kernel weights 
w  = k_weights_outer[0,ix]/sum(k_weights_outer[0,ix])
kd_list_selected = []
kc_list_selected = []
for i_p in range(len(w)):
    kd_list_selected.append(kd_list[kd_ids[ix[i_p]]])
    kc_list_selected.append(kc_list[kc_ids[ix[i_p]]])
    
# Inner CV loop
rmse_inner = np.empty([3, len(regparam)]); rmse_inner[:] = np.NAN

# Read pre-defined inner folds used in the experiments presented in pairwiseMKL paper
inner_folds = np.loadtxt(data_path + '/Folds/inner_folds_outer%d.txt'%i_out).astype(int)
    
for i_in in range(3):
    print('    Inner loop ' + str(i_in+1))
    
    inner_test_ids  = np.array(np.where(inner_folds==i_in)).squeeze()
    inner_train_ids = np.array(np.where((inner_folds!=i_in) & (inner_folds!=-1))).squeeze()

    y_test_inner        = y_vec_known[inner_test_ids]
    drug_ids_test_inner = drug_ids_known[inner_test_ids]
    cell_ids_test_inner = cell_ids_known[inner_test_ids]
    
    y_train_inner        = y_vec_known[inner_train_ids]
    drug_ids_train_inner = drug_ids_known[inner_train_ids]
    cell_ids_train_inner = cell_ids_known[inner_train_ids]

    # Find optimal \lambda
    for i_param in range(len(regparam)):
        # Training           
        learner = CGKronRLS(K1 = kd_list_selected, 
                            K2 = kc_list_selected, 
                            weights = w.tolist(),
                            Y = y_train_inner, 
                            label_row_inds = [drug_ids_train_inner for i in range(len(w))], 
                            label_col_inds = [cell_ids_train_inner for i in range(len(w))], 
                            regparam = regparam[i_param], 
                            maxiter = 400)
        # Prediction 
        pred_inner = learner.predict(kd_list_selected, kc_list_selected, [drug_ids_test_inner for i in range(len(w))], [cell_ids_test_inner for i in range(len(w))])
        # RMSE
        rmse_inner[i_in,i_param] = sqrt(((y_test_inner - pred_inner) ** 2).mean(axis=0))
   
# \lambda with the lowest RMSE 
model = regparam[np.argmin(np.mean(rmse_inner, axis=0))]
        

# Model training with selected \lambda       
learner = CGKronRLS(K1 = kd_list_selected, 
                    K2 = kc_list_selected, 
                    weights = w.tolist(),
                    Y = y_train, 
                    label_row_inds = [drug_ids_train for i in range(len(w))], 
                    label_col_inds = [cell_ids_train for i in range(len(w))], 
                    regparam = model, 
                    maxiter = 400)
# Prediction
y_pred_outer_vec[test_ids] = learner.predict(kd_list_selected, kc_list_selected, [drug_ids_test for i in range(len(w))], [cell_ids_test for i in range(len(w))])


# RMSE
rmse_outer    = sqrt(((y_test - y_pred_outer_vec[test_ids]) ** 2).mean(axis=0))
# Pearson correlation
pearson_outer = np.corrcoef(y_test, y_pred_outer_vec[test_ids])[0,1]
# F1 score
y_test_binary = copy.deepcopy(y_test)
y_test_binary = preprocessing.binarize(y_test_binary.reshape(1,-1), threshold=5, copy=False)[0]
y_pred_binary = copy.deepcopy(y_pred_outer_vec[test_ids])
y_pred_binary = preprocessing.binarize(y_pred_binary.reshape(1,-1), threshold=5, copy=False)[0]
f1_outer      = metrics.f1_score(y_test_binary, y_pred_binary)



# Predicted drug responses
np.savetxt(new_path + '/y_pred_vec__fold_' +  str(i_out+1) + '.txt', y_pred_outer_vec, delimiter='\t')  
# Test ids
np.savetxt(new_path + '/test_ids__fold_' +  str(i_out+1) + '.txt', test_ids, delimiter='\t')  

# In the below files, each row corresponds to the result from a single outer CV fold   
# RMSE
np.savetxt(new_path + '/RMSE__fold_' +  str(i_out+1) + '.txt', np.asarray([rmse_outer])) 
# Pearson correlation
np.savetxt(new_path + '/Pearson_correlation__fold_' +  str(i_out+1) + '.txt', np.asarray([pearson_outer]))  
# F1 score
np.savetxt(new_path + '/F1_score__fold_' +  str(i_out+1) + '.txt', np.asarray([f1_outer]))  
# Optimal value of the regularization parameter \lambda 
np.savetxt(new_path + '/selected_lambda__fold_' +  str(i_out+1) + '.txt', np.asarray([model]))
# Pairwise kernel weights
np.savetxt(new_path + '/pairwise_kernel_weights__fold_' +  str(i_out+1) + '.txt', k_weights_outer, delimiter='\t')  
# Vector a
np.savetxt(new_path +'/a__fold_' +  str(i_out+1) + '.txt', a, delimiter='\t')

# File names of the corresponding pairwise kernels, in the same order as in the file
# with pairwise kernel weights 
if i_out==0:
    thefile = open(new_path + '/pairwise_kernel_names.txt', 'w')
    for i in range(P):
        thefile.write("%s_KRONECKER_%s\t" %(kd_file_names[kd_ids[i]], kc_file_names[kc_ids[i]]))
    thefile.close()


print('\nSuccess!') 