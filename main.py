import numpy as np
import copy
from math import sqrt
from sklearn import preprocessing, metrics
from pairwisemkl.learner.compute_M import *
from pairwisemkl.learner.compute_a_regression import *
from pairwisemkl.learner.optimize_kernel_weights import *
from pairwisemkl.learner.cg_kron_rls import CGKronRLS


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


# Number of pairwise kernels 
P = len(kd_list)*len(kc_list)
# Generate pairwise kernel ids
kd_ids, kc_ids = np.unravel_index(np.arange(P), (len(kd_list),len(kc_list)), order = 'C')


# Compute matrix M needed to optimize pairwise kernel weights
M = compute_M(kd_list, kc_list)


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

# Pairwise kernel weights from each outer fold
k_weights_outer = np.empty([10,P]); k_weights_outer[:] = np.NAN
# Selected value for the regularization parameter \lambda
model = np.empty([10,1]); model[:] = np.NAN
# Predicted drug responses
y_pred_outer_vec = np.empty([len(ids[ids_known])]); y_pred_outer_vec[:] = np.NAN
# Root mean squared error (RMSE) 
rmse_outer = np.empty([10, 1]); rmse_outer[:] = np.NAN
# Pearson correlation
pearson_outer = np.empty([10, 1]); pearson_outer[:] = np.NAN
# F1 score
f1_outer = np.empty([10, 1]); f1_outer[:] = np.NAN

# Read pre-defined outer folds used in the experiments presented in pairwiseMKL paper
outer_folds = np.loadtxt(data_path + '/Folds/outer_folds.txt').astype(int)

# Outer CV loop
for i_out in range(10):
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
    k_weights_outer[i_out,:] = optimize_kernel_weights(a, M)
    

    # Stage 2 of pairwise model training 
    
    # Find pairwise kernels with weights different from 0 (>10**-3).
    ix = np.where(k_weights_outer[i_out,:] > 10**-3)[0]
    # Corresponding kernel weights 
    w  = k_weights_outer[i_out,ix]/sum(k_weights_outer[i_out,ix])
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
    model[i_out,0] = regparam[np.argmin(np.mean(rmse_inner, axis=0))]
            
    
    # Model training with selected \lambda       
    learner = CGKronRLS(K1 = kd_list_selected, 
                        K2 = kc_list_selected, 
                        weights = w.tolist(),
                        Y = y_train, 
                        label_row_inds = [drug_ids_train for i in range(len(w))], 
                        label_col_inds = [cell_ids_train for i in range(len(w))], 
                        regparam = model[i_out,0], 
                        maxiter = 400)
    # Prediction
    y_pred_outer_vec[test_ids] = learner.predict(kd_list_selected, kc_list_selected, [drug_ids_test for i in range(len(w))], [cell_ids_test for i in range(len(w))])


    # RMSE
    rmse_outer[i_out]    = sqrt(((y_test - y_pred_outer_vec[test_ids]) ** 2).mean(axis=0))
    # Pearson correlation
    pearson_outer[i_out] = np.corrcoef(y_test, y_pred_outer_vec[test_ids])[0,1]
    # F1 score
    y_test_binary = copy.deepcopy(y_test)
    y_test_binary = preprocessing.binarize(y_test_binary.reshape(1,-1), threshold=5, copy=False)[0]
    y_pred_binary = copy.deepcopy(y_pred_outer_vec[test_ids])
    y_pred_binary = preprocessing.binarize(y_pred_binary.reshape(1,-1), threshold=5, copy=False)[0]
    f1_outer[i_out] = metrics.f1_score(y_test_binary, y_pred_binary)
    


# Predicted drug responses
np.savetxt(data_path + '/y_pred_vec.txt', y_pred_outer_vec, delimiter='\t')  

# In the below files, each row corresponds to the result from a single outer CV fold   
# RMSE
np.savetxt(data_path + '/RMSE.txt', rmse_outer, delimiter='\t')  
# Pearson correlation
np.savetxt(data_path + '/Pearson_correlation.txt', pearson_outer, delimiter='\t')  
# F1 score
np.savetxt(data_path + '/F1_score.txt', f1_outer, delimiter='\t')  
# Optimal value of the regularization parameter \lambda 
np.savetxt(data_path + '/selected_lambda.txt', model, delimiter='\t')
# Pairwise kernel weights
np.savetxt(data_path + '/pairwise_kernel_weights.txt', k_weights_outer, delimiter='\t')  

# File names of the corresponding pairwise kernels, in the same order as in the file
# with pairwise kernel weights 
thefile = open(data_path + '/pairwise_kernel_names.txt', 'w')
for i in range(P):
    thefile.write("%s_KRONECKER_%s\t" %(kd_file_names[kd_ids[i]], kc_file_names[kc_ids[i]]))
thefile.close()


print('\nSuccess!') 