function A = Gen_datadep_GTmatrix2( H, c)
%% function  A = Gen_datadep_GTmatrix(Y, m, c)
% Generates a group testing matrix A with m tests based on the label matrix Y 
% Input : 
% Y - Label matrix
% m - number of tests (rows in A)
% c - number of nonzeros per column

%% ---
[d,~]=size(H);

%[U, D] = eigs(Y*Y',m);  %%  compute low rank decompostion for potential func.
%U=U*real(sqrt(D));

%% Form the matrix
A = [];
for i=1:d
    
    Pi = H(i,:)'./sum(H(i,:));  % potential function for sampling
    
    vec = Gen_sprand_vec(Pi, c);
    
    A = [A, vec];
end