function vec = Gen_sprand_vec(Pi, c)

n=length(Pi);

% Pi = c*Pi;
% id1=Pi>1;
% exc = sum(Pi(id1) -1);
% id2=find(Pi<0.5);
% exc=exc/length(id2);
% Pi(id2)= Pi(id2)+exc;

Pi = reweight_probvec(Pi, c);

vec=zeros(n,1);
for j=1:n

    prob_j = min([1 Pi(j)]);  % find the minimum of 1 and  c*pi(j)
    prob_j = prob_j(1);         % resolve the case where 1 = c*pi(j)
    
    if prob_j==1             % if prob_j=1 select the j-th entry
       vec(j)=1;
    elseif  prob_j > rand    % if prob_j<1, generate a random number rand in [0,1] and then 
       vec(j)=1;       % if prob_j > rand, select the j-th column of A 
    end
    
end

vec=sparse(vec);
 