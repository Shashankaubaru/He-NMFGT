# MLGT
Matlab code for Multilabel Classification by Hierarchical Partitioning and Data-dependent Grouping


Run main_NMFGT.m for mutlilabel classification by random constant weight and data dependent group testing approaches

Run main_HeNMFGT.m for  mutlilabel classification by label Partitioning and apply NMF_MLGT individually and combine results.

Use main_HeNMFGT_large.m for the larger datasets.

Sel_c_gen_data_GTmatrix - creates data dependent group testing matrix

Sel_c_k_disjunct - creates random constant weight group testing matrix

MLGT_train_test.m contains the training and testing routines of MLGT


Data subfolder contains 4 datasets.

Reordering subfolder contains the permutation (*-perm.txt) for the reordering of labels, and the partition of the label set into subsets (*-comm.txt). The FORTRAN library for matrix reordering is also included.


