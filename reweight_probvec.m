function Pi = reweight_probvec(Pi, c)


n=length(Pi);

Pi = c*Pi;
id1=Pi>=1;
%exc = sum(Pi(id1) -1);
exc=0;
%Pi(id1)=1;
if (sum(id1) >=c)
    return;
else
    [tmp,id2] = sort(Pi, 'descend');
    for j =1:n
        if(tmp(j)>=1)
           exc = exc+ (tmp(j)-1);
           tmp(j)=1;
        elseif (tmp(j)==0)
            tmp(j:n) = tmp(j:n)+exc/(n-j);
            exc=0;
        else
            tmp(j:n) = tmp(j:n)*(1 +exc/sum(tmp(j:n))); 
            %exc = exc*(1 - tmp(j)/ sum(tmp(j:n)));
            exc = 0;
            if(tmp(j)>=1)
                exc = exc+ (tmp(j)-1);
                tmp(j)=1;
            end
        end
        if (sum(tmp==1)>=c || exc == 0)
           Pi(id2) = tmp;
        return;
        end
    end
  Pi(id2) = tmp;       
end
