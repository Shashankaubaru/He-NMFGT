function [A, c, er] = Sel_c_gen_data_GTmatrix(Y, m, n,k, c1, options)
%% function[A, c]  = Sel_c_gen_data_GTmatrix(Y, m, n,k, c1, options)
% This function generates the data dependent group testing matrix A using
% (symmetric) NMF of the training label matrix Y.

%%--- Inputs
% Y - Training label matrix
% m - Number of groups (rows in A) needed
% n - Number of training instances to use
% k = Label sparsity
% c1 = column sparsity sweep
% options - NMF paramters

%%-- Outputs
% A - Group testing matrix
% c - column sparsity
% er - Hamming loss error in recovery

%%%%

%% Get NMF
[d,~]=size(Y);

 [H, ~, ~] = symNMF(Y*Y',m, options);
% H = repmat(H1,  1, m/100);
%[H,~] = nnmf(Y,m,'opt',options,'algo','mult');
%c1= 5:5:30;

%% Select c and A
for t=1:length(c1)

        A = Gen_datadep_GTmatrix2( H, c1(t)); 
        
        Z=spones(A*Y(:,1:n));
        
         ATp=A'*Z;
        err=zeros(n,1);
        for l=1:n
            yp=zeros(d,1);
            [~,idx]=sort(ATp(:,l),'descend');
            yp(idx(1:k),1)=1;
            err(l,1)=sum(yp~=Y(:,l));
        end
        Err(t,1)=mean(err);
        Atmp{t} = A; 
end
[er,idx] = min(Err);
A = Atmp{idx};
c = c1(idx);
