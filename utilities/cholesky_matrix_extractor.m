% Feb/Mar2021: Shaked: iterativedirect_alpha.m
%              implements Algorithm 1 in the KKT report.
%              Reads data and runs combined iterative/direct method
%              on a series of matrices from OPF interior method.
% 04 Mar 2021: Michael: reformatted.

clear all;
scale = 1; %1 for scaling, 0 for no scaling
exactnorm = 0; %1 for the exact norm, 0 to estimate with random vector
start =  0;  % first index in sequence
fin   = 0;  % last  index in sequence
nexp  = fin-start+1;
buf   = [];

% matrix and vector prefixes
matpref = "matrix_ACTIVSg200_AC_"; %set this to the desired matrix prefix
vecpref = "rhs_ACTIVSg200_AC_"; %set this to the desired vector prefix

% Equivalent modification (gamma) parameter
expon = 6;
nex   = length(expon);

for kk=start:fin  %first to last matrix in the sequence
    if kk<10
       buf='0';
    else
       buf=[];
    end
    matname = strcat(matpref , buf, num2str(kk), ".mtx");
    vecname = strcat(vecpref    , buf, num2str(kk), ".mtx");
    
    [A, rows , cols , entries ] = mmread (matname);
    [b, rows2, cols2, entries2] = mmread (vecname);
    bnorm = norm(b);
    %%
    if kk==start  % for first matrix only, find structure and reuse it
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
       rvec  = randn(top,1);
       rvecn = rvec/norm(rvec);  % to estimate norms
    end

    %% Grab the relevant blocks
    for ind=1:nex
       if ind==1
          H   = A(1:top,1:top);  % this includes Dx
          Dd  = A(top+1:mid,top+1:mid);
          Jc1 = A(mid+1:end-dsize,1:top);
          Jd  = A(end-dsize+1:end,1:top);
          rd  = b(top+1:mid);
          ryd = b(rows-dsize+1:rows);

          %% Make it 2X2
          b1  = [b(1:top) + Jd'*(Dd*ryd+rd)
                 b(mid+1:rows-dsize)];
          b1norm = norm(b1);
          addblock = Jd'*Dd*Jd;

          % Force symmetry.  There's probably a better way
          block11    = H + (tril(addblock) + tril(addblock,-1)');
          A1         = [block11  Jc1'
              Jc1      sparse(csize,csize)];

          if (scale)
            [A2,b2,D2] = SymRuizScaling(A1,b1);
          else
              A2 = A1; b2 = b1; D2 = speye(length(A1));
          end
          block11    = A2(1:top,1:top);
          Jc1        = A2(top+1:end,1:top);
       end

        %% (1,1) block and rhs modification
        alpha      = 10^expon(ind);
        block11mod =    block11 + alpha*(Jc1'*Jc1);
        bmod       = [b2(1:top) + alpha*(Jc1'*b2(top+1:end))
                      b2(top+1:end)];

        % Reorder the system using symmetric approximate minimum degree
        if ind==1 && kk==start  % only done once because AMD depends on structure, not values
            p     = symamd(block11mod);
          % ip(p) = 1:top;   % Not needed
        end
        Htilde = block11mod(p,p);
        Ht     = Htilde; %line 4 in Alg 1
        gtil   = bmod(p);
        matname = strcat("H_SPD_", matpref, buf, num2str(kk),".mtx");
        vecname = strcat("g_", vecpref, buf, num2str(kk),".mtx");
        mmwrite(matname,Ht);
        mmwrite(vecname,gtil);
    end
end
