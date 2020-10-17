function Output  = MLGT_test(SVM, A, Xtest)
%% function Output  = MLGT_test(SVM, A, Xtest)
% This function contains the main  test routine for MLGT

%%--- Inputs
% SVM - The m SVM classifiers previously trained
% Xtest - Test feature matrix
% A - Group testing matrix

%%-- Outputs
% Output.ATp  = The score functions (inner products)
% Output.test_time  = Testing time (cputime)
%%% ---
%addpath(genpath('XMLPref_eval'))
%% -- Initialization
[m,d]=size(A);
%[n,~]=size(X);
[nt,~]=size(Xtest);
Ztest = zeros(m,nt);
%% Testing
 t2 = cputime;
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
Output.test_time = t3 - t2;
