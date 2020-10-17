clc
close all
%clear all

addpath 'XMLDatasetRead'
addpath 'llst/matlab_code'
addpath(genpath('XMLPerf_eval'))

%% Load data

% [X,Y]=read_data('RCV1-x/rcv1x_train.txt');
% [Xtest,Ytest]=read_data('RCV1-x/rcv1x_test.txt');

load('RCV1x.mat');

%% Set parameters

% Get sizes
[p,n]=size(X');    
[d,~]=size(Y);
[p,nt]=size(Xtest');

% MLGT parameters
c1 = 10:10:70;  % column sparsity sweep
m= 250;         % Number of groups
k=5;            % Number of labels per instance
%SymNMF parameters
options.maxiter         = 200;    % Maximum number of iterations
options.timelimit       = 60*3;      % Maximum time of execution
%options = statset('maxiter',100,'display','final');

%%  NMF (data-dependent MLGT)
[A1, c, Err1] = Sel_c_gen_data_GTmatrix(Y, m, n,k, c1, options);

 Output1  =  MLGT_train_test_prec(X, Y, Xtest, Ytest,A1, k);
%%% ---

%% CW (Constant Weight MLGT)

[A2, c, Err2] = Sel_c_k_disjunct(Y, m, n,k, c1);

Output2  =  MLGT_train_test_prec(X, Y, Xtest, Ytest,A2, k);

%% Get results

NMF_GT_Prec = [Output1.Prec_k(1),  Output1.Prec_k(3) Output1.Prec_k(5)];
NMF_times = [Output1.train_time, Output1.test_time];

CW_GT_Prec = [Output2.Prec_k(1), Output2.Prec_k(3), Output2.Prec_k(5)];
CW_times = [Output2.train_time, Output2.test_time];

%% 

