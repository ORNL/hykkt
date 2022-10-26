function [mid,top,dsize,csize] = findblocks(A,rows)
       dA      = diag(A);
       [i,j,v] = find(dA);
       mid     = i(end);  % the end of the (2,2) block
       block21 = A(mid+1:end,1:mid);
       [i,j,v] = find(block21);
       diff    = j(end)-i(end);
       flag    = 1;
       ind     = length(i);
       
       while flag==1
           if (j(ind)-i(ind)~=diff), break, end
           ind = ind-1;
       end
       top   = j(ind);
       dsize = mid-top;
       csize = rows-dsize-mid;
end