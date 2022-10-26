function [As, bs, D2] = SymRuizScaling(A,b)
% Author: Shaked Regev
% Purpose: implement iterative scaling according to Ruiz.
% You need D2 for updating solution
As = tril(A);
bs = b;
Rconv = 0;
it =0;
n = length(A);
e = ones(n,1);
D2 = speye(n);
while ((Rconv == 0))
maxr=sqrt(max(abs(As), [], 2));
maxc=sqrt(max(abs(As), [], 1));
maxt=max(maxr,maxc');
Dr = diag(maxt);

As = Dr\As/Dr;
bs = Dr\bs;
D2 = D2/Dr;

it = it+1;
%keyboard
if (max(abs(maxt.^2 - e))<1e-1 || it>1)
    Rconv = 1;
    disp(it);
    As=As+tril(As,-1)';
end
end