function P = precision_k_new(score_mat,true_mat,K)
P = helper(score_mat,true_mat,K);
end

function P = helper(score_mat,true_mat,K)
num_inst = size(score_mat,2);
num_lbl = size(score_mat,1);

P = zeros(K,1);
score_mat = sparse(score_mat);
rank_mat = sort_sparse_mat(score_mat);

mat = rank_mat;
mat =[];
for j=1:num_inst
    tmp = rank_mat(:,j);
    tmp(tmp>K) =0;
    mat = [mat spones(tmp)];
end
mat = spones(mat);
mat = mat.*true_mat;
num = sum(mat,1);

for k=1:K
    num2 = num;
    num2(num2>=k)=k;
    P(k) = mean(num2/k);
end
end
