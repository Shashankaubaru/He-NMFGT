clc
close all
%clear all

addpath 'XMLDatasetRead'
addpath(genpath('XMLPerf_eval'))

%% Load data

 [X,Y]=read_data('Amazon/amazon_train.txt');
 [Xtest,Ytest]=read_data('Amazon/amazon_test.txt');
 X= X'; Xtest = Xtest'; 
%%--- Use above if data is from the XML repository
%% Set parameters

% Get sizes
[p,n]=size(X');    
[d,~]=size(Y);
[p,nt]=size(Xtest');

% MLGT parameters
c1 = 10:10:70;  % column sparsity sweep
m= 800;         % Number of groups
k=5;            % Number of labels per instance
%SymNMF parameters
options.maxiter         = 200;    % Maximum number of iterations
options.timelimit       = 60*3;      % Maximum time of execution
%options = statset('maxiter',100,'display','final');


%% Matrix reordering and partitioning
% Read the permutation obtained
filename = 'Reordering/Amazon-perm.txt'; 
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
set = {  [1:21508, 628135:628211, 629900:631044, 633426:636359, 663137:670091],...
% Fill in the rest
        };
    
 n_par = length(set); % N o. of partitions
 
 for i =1: n_par
     Yp{i} = Y(set{i},:);
     Ytp{i}= Ytest(set{i},:);
 end
    
%% Perform MLGT training on each set (can be done in parallel)

ATp=  sparse(d,nt);
for i =1: n_par
    Y1 = Yp{i}; X1 = X;
    
    %%Negative sampling of training data
    idx = ~any(Y1,1);
    Y1(:,idx)=[]; X1(:,idx)=[];

    [A1, ~, ~] = Sel_c_gen_data_GTmatrix(Y1, m, n,k, c1, options);

    Output1  =  MLGT_train(X1, Y1,A1);
    SVM{i} =  Output1.SVM; 
    GT_A{i} = A1;
end

%% Save the trained model and GT matrices

%save('Amazon_train.mat', 'SVM','GT_A','-v7.3');

%% MLGT testing. 
%%-- For large datasets, we will have to test one/few instance/s at a time
P = zeros(nt,1);
for l = 1:nt
ATp=  sparse(d,1); 
for i =1:n_par
        Xt = Xtest(l,:);
        Output1 = MLGT_test(SVM{i}, GT_A{i}, Xt, k);
        tmp_ATp =  zeros(d,1);
        tmp_ATp(set{i},:) =  Output1.ATp;
   ATp = ATp + tmp_ATp; 
end
Yt = Ytest(:,l);
P(l,1) = precision_k(sparse(ATp),Yt,k);
end
P = mean(P);
%% Get results

fprintf('HeNMFGT Precision at k = 1,3,5 are %f, %,f %f', P(1), P(3), P(5));

%% 

