clc
close all
%clear all

addpath 'XMLDatasetRead'
addpath(genpath('XMLPerf_eval'))

%% Load data

%[X,Y]=read_data('Eurlex/eurlex_train.txt');
% [Xtest,Ytest]=read_data('Eurlex/eurlex_test.txt');
%%--- Use above if data is from the XML repository

load('Data/Eurlex.mat');  

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


%% Matrix reordering and partitioning
% Read the permutation obtained
filename = 'Reordering/Eulrex-perm.txt'; 
fid =fopen(filename, 'r');
r = fscanf(fid, '%d');
fclose(fid); 
%%-- Permute the labels
S = Y*Y';               % Compute the correlation matrix
%r = symamd(S);          % Obtain the reodering by AMD algorithm         
S = S(r,r);             % Reorder the labels
Y = Y(r,:);
Ytest = Ytest(r,:);

%% Obtain/Define partitions
%%--Get the set partitions from Reordering/*-comm.txt 
set = { [1:992,   2571:d],...
        [993:1913,   2571:d],...
        [1914:d]...
        };
    
 n_par = length(set); % N o. of partitions
 
 for i =1: n_par
     Yp{i} = Y(set{i},:);
     Ytp{i}= Ytest(set{i},:);
 end
    
%% Perform MLGT on each set (can be done in parallel)

ATp=  sparse(d,nt);
for i =1: n_par
    Y1 = Yp{i}; X1 = X;
    
    %%Negative sampling of training data
    idx = ~any(Y1,1);
    Y1(:,idx)=[]; X1(:,idx)=[];

    [A1, ~, ~] = Sel_c_gen_data_GTmatrix(Y1, m, n,k, c1, options);

    Output1  =  MLGT_train_test(X1, Y1, Xtest, Ytest,A1, k);
    tmp_ATp =  sparse(d,nt);
    tmp_ATp(set{i},:) =  Output1.ATp;
    
    ATp = ATp + tmp_ATp;   
end

%% Get precision

P_new = precision_k_new(ATp,Ytest,k);  %% Modified Precision
P = precision_k(ATp,Ytest,k);  %% Original Precision

%% Get results

fprintf('HeNMFGT Precision at k = 1,3,5 are %f, %,f %f', P(1), P(3), P(5));

fprintf('HeNMFGT Modified Precision at k = 1,3,5 are %f, %,f %f', P_new(1), P_new(3), P_new(5));

%% 

