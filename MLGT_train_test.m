function Output  = MLGT_train_test(X, Y, Xtest, Ytest,A, k)
%% function Output  = MLGT_train_test(X, Y, Xtest, Ytest,A, k)
% This function contains the main training and test routines for MLGT

%%--- Inputs
% X - Training feature matrix
% Y - Training label matrix
% Xtest - Test feature matrix
% Ytest - Test label matrix
% A - Group testing matrix
% k = Label sparsity

%%-- Outputs
% Output.ATp  = The score functions (inner products)
% Output.train_time  = Training time (cputime)
% Output.test_time  = Testing time (cputime)
%%% ---
%addpath(genpath('XMLPref_eval'))
%% -- Initialization
[m,d]=size(A);
[n,~]=size(X);
[nt,~]=size(Xtest);
Ztest = zeros(m,nt);
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
%% Testing

for l=1:m
    pt=predict(SVM{l},Xtest);
    Ztest(l,:)=pt';
end
Ztest = sparse(Ztest);
ATp=A'*Ztest;
 t3 = cputime;

 %P = precision_k_new(ATp,Ytest,k);
 %N = nDCG_k(ATp,Ytest,k);

%% Get results

%Output.Prec_k=P;
Output.ATp=ATp;
Output.train_time = t2-t1;
Output.test_time = t3 - t2;
Output.total_times = t3-t1;
