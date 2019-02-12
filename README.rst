=======
pairwiseMKL
=======


pairwiseMKL package.


:Authors:         Anna Cichonska, Sandor Szedmak, Tapio Pahikkala, Antti Airola
:Email:           anna.cichonska@helsinki.fi
:Version:         0.1
:License:         `The MIT License <LICENCE.TXT>`_
:Date:            June 2018

.. contents::

Overview
========

pairwiseMKL is a machine learning software package implemented in Python for learning with multiple pairwise kernels.

Folder drug_response_data contains drug response in cancer cell lines dataset used in the pairwiseMKL paper. 
Script main.py demonstrates the usage of pairwiseMKL.

The computations can be easily parallized using array jobs, which is especially useful when working with large datasets. Script main_precalculate_M_arrayjob.py shows how to calculate each row of the matrix M needed to find kernel weights using array jobs. The script takes a row number of the matrix M as input. In case of drug response dataset with 120 pairwise kernels, the corresponding bash script should contain #SBATCH --array=0-119.
Given pre-calculated matrix M, script main_arrayjob_using_precalculated_M.py demonstrates how to run each outer cross validation (CV) loop as a separate array job. This script takes a number identifying the outer CV loop as input. In the pairwiseMKL paper, we used 10x3 nested CV (10 outer folds, 3 inner folds), and thus the corresponding bash script should contain #SBATCH --array=0-9.



Installation
========

Global installation:
python setup.py install

Installing to home directory:
python setup.py install --home=<dir>


pairwiseMKL requires `cvxopt <https://cvxopt.org/>`_ package for convex optimization. 
It can be installed, e.g., by:
conda install -c conda-forge cvxopt

This project is based on python 3. It is recommended to use python 3.5 to use cvxopt package on python 3.



Citing pairwiseMKL
==============

pairwiseMKL is described in the following article:

`Learning with multiple pairwise kernels for drug bioactivity prediction <https://academic.oup.com/bioinformatics/article/34/13/i509/5045738>`_, Anna Cichonska, Tapio Pahikkala, Sandor Szedmak, Heli Julkunen, Antti Airola, Markus Heinonen, Tero Aittokallio, Juho Rousu. Bioinformatics, 34(13):i509-i518, 2018.


