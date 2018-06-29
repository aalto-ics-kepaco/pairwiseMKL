#
# The MIT License (MIT)
#
# This file is part of pairwiseMKL
#
# Copyright (c) 2018 Anna Cichonska, Sandor Szedmak
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


def kron_decomp_centralization_operator(m,n):
  """
  Task: to compute the factors of the pairwise kernel centralization operator 
        with dimension mn=m*n:
        C_{mn} = I_{mn} - 1_{mn} \otimes 1'_{mn} / mn

        I_{mn}=np.eye(mn)
        1_{mn}=np.ones(mn)

        C_nm reproduced as 
        C_mn= Q[0][0] \otimes Q[0][1] + Q[1][0] \otimes Q[1][1]

        The factors have the structure:
        Q[0][0]=(w_{000}-w_{001}) I_{m} + w_{001} 1_m \otimes 1'_m
        Q[0][1]=(w_{010}-w_{011}) I_{n} + w_{011} 1_n \otimes 1'_n
        Q[1][0]=(w_{100}-w_{101}) I_{m} + w_{101} 1_m \otimes 1'_m
        Q[1][1]=(w_{110}-w_{111}) I_{n} + w_{111} 1_n \otimes 1'_n
  
  Input:   m      The size m\times m of the first factor
           n      The size n\times n of the second factor
  
  Output:  Q      List of lists, 2\times 2,  of the factor matrices:
                  C_mn = Q[0][0] \otimes Q[0][1] + Q[1][0] \otimes Q[1][1]
                  
  References:
  [1] Anna Cichonska, Tapio Pahikkala, Sandor Szedmak, Heli Julkunen, Antti Airola, 
  Markus Heinonen, Tero Aittokallio, Juho Rousu.
  Learning with multiple pairwise kernels for drug bioactivity prediction.
  Bioinformatics, 34, pages i509–i518. 2018.
  """

  # Two singular values, two factors, two weights
  nsvalue = 2
  nfactor = 2
  nweight = 2
  xw = np.zeros((nsvalue,nfactor,nweight)) # the component weights
  mn = m*n  # the full size of the Kronecker product matrix

  # The compressed reordered centralization matrix
  Q = np.array([[mn-1,-(n-1)],[-(m-1),-(m-1)*(n-1)]])
  # The singular vectors are rescaled for the compressed matrix 
  qu = np.array([1/m**0.5,1/(m*(m-1))**0.5])
  qv = np.array([1/n**0.5,1/(n*(n-1))**0.5])
  Creduced = Q*np.outer(qu,qv)

  # Singular value decomposition of the compressed matrix
  (Ur,Sr,Vr) = np.linalg.svd(Creduced)
  # Vr is provided as transpose by numpy linalg  
  Vr = Vr.T

  # Recover the components of the singular vectors
  # of the original uncom,pressed matrix
  U = Ur*np.outer(qu,np.ones(nsvalue))
  V = Vr*np.outer(qv,np.ones(nsvalue))
  # Recover the singular values for the uncompressed matrix
  singval = np.diag(np.dot(U.T,np.dot(Q,V)))
  # print(singval)
  # Compute the weights:
  # components of the singular vectors * sqrt(singular values)
  Uw = U*np.outer(np.ones(nsvalue),np.sqrt(singval))
  Vw = V*np.outer(np.ones(nsvalue),np.sqrt(singval))

  # The weight matrix
  xw[0] = np.vstack((Uw[:,0],Vw[:,0]))
  xw[1] = np.vstack((Uw[:,1],Vw[:,1]))

  # Build the factors from the weights
  Qfactors = [[None,None] for _ in range(nsvalue)]
  factorsize = [m,n]
  for i in range(nsvalue):
    for j in range(nfactor):
      Qfactors[i][j] = (xw[i,j,0]-xw[i,j,1])*np.eye(factorsize[j]) \
        +xw[i,j,1]*np.ones((factorsize[j],factorsize[j]))
  
  return Qfactors