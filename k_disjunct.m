function A = k_disjunct(m, d, p1)

%% This function creates a k disjunct constant weight group testing matrix


A2 = sparse(m,d);


rep1=floor(d/p1);
rep2=ceil(m/rep1);
j1=1;
for j2=1:rep2
    if(j2==1)
        for j=1:rep1
            A2(j,j1:j1+p1-1)=1;
            j1=j1+p1;
        end
    else
        rn=randperm(d);
        for j=1:rep1
            A2((j2-1)*rep1+j,:)=A2(j,rn);
        end
    end
end
A2=A2(1:m,:);
rd=randperm(d);
 A = A2(:,rd);


