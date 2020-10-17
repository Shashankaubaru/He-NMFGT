function [A, c, er] = Sel_c_k_disjunct(Y, m, n,k, c1)
%% function[A, c, er]  =  Sel_c_k_disjunct(Y, m, n,k, c1)
% This function generates the constant weight group testing matrix A 

%%--- Inputs
% Y - Training label matrix
% m - Number of groups (rows in A) needed
% n - Number of training instances to use
% k = Label sparsity
% c1 = column sparsity sweep

%%-- Outputs
% A - Group testing matrix
% c - column sparsity
% er - Hamming loss error in recovery

%%%%

%% Get parameter
[d,~]=size(Y);

%c1= 5:5:30;

%% Select c and A
for t=1:length(c1)

        A =  k_disjunct(m, d, floor(c1(t)*d/m));
        
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
