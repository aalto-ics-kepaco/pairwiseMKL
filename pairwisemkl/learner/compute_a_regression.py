#
# The MIT License (MIT)
#
# This file is part of pairwiseMKL
#
# Copyright (c) 2018 Anna Cichonska
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import math
from scipy import stats
from pairwisemkl.learner.kron_decomp import kron_decomp_centralization_operator
    

def response_kernel_features(Y):
    """ 
    Task:    to compute feature vector for each label value
  
    Input:   Y      Matrix with the original labels
  
    Output:  Psi_y  Matrix storing features as row vectors    

    References:
    [1] Anna Cichonska, Tapio Pahikkala, Sandor Szedmak, Heli Julkunen, Antti Airola, 
    Markus Heinonen, Tero Aittokallio, Juho Rousu.
    Learning with multiple pairwise kernels for drug bioactivity prediction.
    Bioinformatics, 34, pages i509–i518. 2018.        
    """
    
    # Labels in the vector form
    y = Y.ravel(order = 'C')   
    
    # Generate probability density function of the labels
    min_y = min(y)
    max_y = max(y)
    n_interv = 50
    step = float(max_y-min_y)/n_interv
    x_interv = np.arange(math.floor((min_y)*10)/10-(n_interv+1)*step, math.ceil((max_y)*10)/10+(n_interv+1)*step, step)
    # Intervals: [x_interv[0],x_interv[1]), [x_interv[1],x_interv[2]), ...
    x = [(a+b)/2 for a,b in zip(x_interv[::1], x_interv[1::1])]  
    kde = stats.gaussian_kde(y)
    x_kde = kde(x)
    # plt.plot(x,x_kde)
    # plt.xlim([min(x),max(x)])
    # plt.show() 

    # Matrix storing features as row vectors (one feature vector per label)
    Psi_y = np.empty([len(y), n_interv*2]); Psi_y[:] = np.NAN  
    for i in range(len(y)):  
        id_i  = np.where(x >= y[i])[0][0]
        Psi_y[i,] = x_kde[id_i-n_interv:id_i+n_interv] 
        Psi_y[i,] = Psi_y[i,]/np.linalg.norm(Psi_y[i,])
    # Ky = Sum_q(Psi_y[:,q] Psi_y[:,q]^T)
    
    return Psi_y



def compute_a_regression(Ka_list, Kb_list, Y):
    """ 
    Task: to compute vector 'a' needed for optimizing pairwise kernel weights 
          (equation 16 of the paper describing pairwiseMKL method)
  
    Input:   Ka_list      List of drug (view A in general) kernel matrices
             Kb_list      List of cell line (view B in general) kernel matrices
             Y            Matrix with the original labels
  
    Output:  a            Vector storing Frobenius inner products between each 
                          centered input pairwise kernel and the response 
                          kernel     
    """
    
    # To compute the factors of the pairwise kernel centering operator 
    Q = kron_decomp_centralization_operator(Ka_list[0].shape[0], Kb_list[0].shape[0])

    # Total number of pairwise kernels
    p = len(Ka_list)*len(Kb_list)   

    ids_kernels    = np.arange(p)
    Ka_ids, Kb_ids = np.unravel_index(ids_kernels, (len(Ka_list),len(Kb_list)), order = 'C')

    # Replace missing values in the label matrix with row means
    if np.isnan(Y).any() == True:
        nan_ids    = np.where(np.isnan(Y))     
        row_mean   = np.nanmean(Y, axis=1)
        Y[nan_ids] = np.take(row_mean,nan_ids[0])
    # If all the values in a row are missing, use global mean
    if np.isnan(Y).any() == True:
        nan_ids_remaining = np.where(np.isnan(Y))
        global_mean = np.nanmean(Y.ravel(order = 'C'))
        Y[nan_ids_remaining] = global_mean
        
    # Compute feature vectors for each label value  
    Psi_y = response_kernel_features(Y)
        
    a = np.zeros([1,p])
    n_y = Psi_y.shape[0]
    
    # Response kernel Ky    
    # K = np.zeros([n_y,n_y])
    # q = 0
    # while q < Psi_y.shape[1]:
    #    v_q = Psi_y[:,q].reshape(n_y,1)
    #    K = K + np.dot(v_q , v_q.T)
    #    q = q + 1
    
    # Calculate elements of the vector 'a'
    for i_pairwise_k in range(p):
        
        i = Ka_ids[i_pairwise_k]
        j = Kb_ids[i_pairwise_k]

        Ka_1 = np.dot( np.dot(Q[0][0],Ka_list[i]), Q[0][0] )
        Ka_2 = np.dot( np.dot(Q[1][0],Ka_list[i]), Q[1][0] )
        Ka_3 = np.dot( np.dot(Q[0][0],Ka_list[i]), Q[1][0] )
        Ka_4 = np.dot( np.dot(Q[1][0],Ka_list[i]), Q[0][0] )

        Kb_1 = np.dot( np.dot(Q[0][1],Kb_list[j]), Q[0][1] )
        Kb_2 = np.dot( np.dot(Q[1][1],Kb_list[j]), Q[1][1] )
        Kb_3 = np.dot( np.dot(Q[1][1],Kb_list[j]), Q[0][1] )
        Kb_4 = np.dot( np.dot(Q[0][1],Kb_list[j]), Q[1][1] )
        
        # Compute  < K_k^(c), K_y^(c)>_F
        q = 0
        while q < Psi_y.shape[1]:
            
            psi_q = Psi_y[:,q].reshape(n_y,1)   # vector
            Psi_q = np.reshape(psi_q, Y.shape, order = 'C') # matrix form
            
            v1 = np.dot( np.dot(Kb_1,Psi_q.T), Ka_1 ).ravel(order = 'F')
            v2 = np.dot( np.dot(Kb_2,Psi_q.T), Ka_2 ).ravel(order = 'F')
            v3 = np.dot( np.dot(Kb_3,Psi_q.T), Ka_3 ).ravel(order = 'F')
            v4 = np.dot( np.dot(Kb_4,Psi_q.T), Ka_4 ).ravel(order = 'F')
        
            a[0, i_pairwise_k] = a[0, i_pairwise_k] + np.dot(psi_q.T, v1+v2+v3+v4)
            q = q + 1
        
    return a 
