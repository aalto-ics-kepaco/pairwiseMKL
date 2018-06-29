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
File main.py demonstrates the usage of pairwiseMKL.



Installation
========

Global installation:
python setup.py install

Installing to home directory:
python setup.py install --home=<dir>


pairwiseMKL requires `cvxopt <https://cvxopt.org/>`_ package for convex optimization. 
It can be installed, e.g., by:
conda install -c conda-forge cvxopt



Citing pairwiseMKL
==============

pairwiseMKL is described in the following article:

`Learning with multiple pairwise kernels for drug bioactivity prediction <https://academic.oup.com/bioinformatics/article/34/13/i509/5045738>`_, Anna Cichonska, Tapio Pahikkala, Sandor Szedmak, Heli Julkunen, Antti Airola, Markus Heinonen, Tero Aittokallio, Juho Rousu. Bioinformatics, 34(13):i509-i518, 2018.


