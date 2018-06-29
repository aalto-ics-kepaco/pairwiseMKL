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
from cvxopt import matrix
from cvxopt import solvers


def optimize_kernel_weights(a, M):
    """ 
    Task:    to determine pairwise kernel weights 
  
    Input:   a      Vector storing Frobenius inner products between each 
                    centered input pairwise kernel and the response kernel  
             M      Matrix storing Frobenius inner products between all pairs 
                    of centered input pairwise kernels
  
    Output:  w      Vector with pairwise kernel weights  
    
    References:
    [1] Anna Cichonska, Tapio Pahikkala, Sandor Szedmak, Heli Julkunen, Antti Airola, 
    Markus Heinonen, Tero Aittokallio, Juho Rousu.
    Learning with multiple pairwise kernels for drug bioactivity prediction.
    Bioinformatics, 34, pages i509–i518. 2018.
    """
 
    n_k = len(M)
    a   = np.array(a,dtype='d').T
    
    P = matrix(2*M)
    q = matrix(-2*a)
    G = matrix(np.diag([-1.0]*n_k))
    h = matrix(np.zeros(n_k,dtype='d'))
    
    sol = solvers.qp(P,q,G,h)
    
    w = sol['x']
    w = w/sum(w)
    
    return np.asarray(w.T)