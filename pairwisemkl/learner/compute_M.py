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
import copy
from pairwisemkl.learner.kron_decomp import kron_decomp_centralization_operator
    

def compute_M(Ka_list, Kb_list):
    """ 
    Task: to compute matrix 'M' needed for optimizing pairwise kernel weights 
          (equation 12 of the paper describing pairwiseMKL method)
  
    Input:   Ka_list      List of drug (view A in general) kernel matrices
             Kb_list      List of cell line (view B in general) kernel matrices
  
    Output:  M            Matrix storing Frobenius inner products between all 
                          pairs of centered input pairwise kernels
                          
    References:
    [1] Anna Cichonska, Tapio Pahikkala, Sandor Szedmak, Heli Julkunen, Antti Airola, 
    Markus Heinonen, Tero Aittokallio, Juho Rousu.
    Learning with multiple pairwise kernels for drug bioactivity prediction.
    Bioinformatics, 34, pages i509–i518. 2018.
    """
    
    # To compute the factors of the pairwise kernel centering operator 
    Q = kron_decomp_centralization_operator(Ka_list[0].shape[0], Kb_list[0].shape[0])

    # Total number of pairwise kernels
    p = len(Ka_list)*len(Kb_list)   
    
    M = np.empty([p,p]); M[:] = np.NAN
    ids_kernels    = np.arange(p)
    Ka_ids, Kb_ids = np.unravel_index(ids_kernels, (len(Ka_list),len(Kb_list)), order = 'C')

    # Calculate elements of the matrix M
    for i_pairwise_k in range(p):

        i = Ka_ids[i_pairwise_k]
        j = Kb_ids[i_pairwise_k]

        h_col_start = i_pairwise_k+1
        h_col_temp  = copy.deepcopy(h_col_start)
        h = 0
        
        for ii in Ka_ids[h_col_start:p]:
            
            jj  = Kb_ids[h_col_start:p][h]
            h   = h+1
            
            # Compute  < K_k, K_l>_F
            M[i_pairwise_k, h_col_temp] = calculate_element(Q, Ka_list[i], Ka_list[ii], Kb_list[j], Kb_list[jj])
            M[h_col_temp, i_pairwise_k] = M[i_pairwise_k, h_col_temp]

            h_col_temp = h_col_temp + 1

        # diagonal(M) =  ( ||K_k||_F )^2
        M[i_pairwise_k, i_pairwise_k] = calculate_element(Q, Ka_list[i], Ka_list[i], Kb_list[j], Kb_list[j])
        
    return M



def calculate_element(Q, Ka_i, Ka_j, Kb_i, Kb_j):
    """ 
    Task: to compute a single element of the matrix M
  
    Input:   Q            List of lists, 2\times 2, of the factor matrices of 
                          the kernel centering operator
             Ka_i         i'th drug kernel matrix
             Ka_j         j'th drug kernel matrix
             Kb_i         i'th cell line kernel matrix
             Kb_j         j'th cell line kernel matrix
  
    Output:  m            Frobenius inner product between centered pairwise 
                          kernels  (Ka_i \otimes Kb_i) and (Ka_j \otimes Kb_j)
    """
    
    nsvalue = 2 
    m = 0
    
    for q in range(nsvalue):
        for r in range(nsvalue):
            m += np.trace( np.dot(np.dot(np.dot(Q[q][0],Ka_i),Q[r][0]),Ka_j) ) \
            * np.trace( np.dot(np.dot(np.dot(Q[q][1],Kb_i),Q[r][1]),Kb_j) )
    
    return m
