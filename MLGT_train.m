function Output  = MLGT_train(X, Y, A)
%% function Output  = MLGT_train(X, Y, A)
% This function contains the main training anf test routines for MLGT

%%--- Inputs
% X - Training feature matrix
% Y - Training label matrix
% A - Group testing matrix

%%-- Outputs
% Output.SVM= The trained SVM classifiers.
% Output.training_time  = Training time (cputime)
%%% ---
%addpath(genpath('XMLPref_eval'))
%% -- Initialization
[m,d]=size(A);
[n,~]=size(X);
%[nt,~]=size(Xtest);
%% Training

t1 = cputime;
Y2=spones(A*Y);   % Label reduction via. Boolean OR

for j=1:m
    y2=Y2(j,:)';

    SVM{j} = fitclinear(X, y2);
       % SVM{j} = fitclinear(X, y2,'Learner','logistic',...
       %                  'Solver','sparsa','Regularization','lasso');
end

t2 = cputime;

%% Get results

%Output.Prec_k=P;
Output.SVM=SVM;
Output.train_time = t2-t1;
