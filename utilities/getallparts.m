clear all;
start =  0;  % first index in sequence
fin   = 0;  % last  index in sequence
matpref = "matrix_ACTIVSg200_AC_"; %set this to the desired matrix prefix
vecpref = "rhs_ACTIVSg200_AC_"; %set this to the desired vector prefix
for kk=start:fin  %first to last matrix in the sequence
    if kk<10
        buf='0';
    else
        buf=[];
    end
    matname = strcat(matpref, buf, num2str(kk), ".mm");
    vecname = strcat(vecpref, buf, num2str(kk), ".mm");
    [A, rows , cols , entries ] = mmread (matname);
    [b, rows2, cols2, entries2] = mmread (vecname);
    if kk==start  % for first matrix only, find structure and reuse it
        [mid,top,dsize,csize] = findblocks(A,rows);
    end
    %matrix blocking
    H   = A(1:top,1:top);  % this includes Dx
    Dd  = A(top+1:mid,top+1:mid);
    Jc1 = A(mid+1:end-dsize,1:top);
    Jd  = A(end-dsize+1:end,1:top);
    rx  = b(1:top);
    rs  = b(top+1:mid);
    ry  = b(mid+1:end-dsize);
    ryd = b(end-dsize+1:end);
    matname = strcat("block_H_", matpref, buf, num2str(kk), ".mtx");
    vecname = strcat("block_rx_", vecpref, buf, num2str(kk), ".mtx");
    mmwrite(matname,H);
    mmwrite(vecname,rx);
    matname = strcat("block_Dd_", matpref, buf, num2str(kk), ".mtx");
    vecname = strcat("block_rs_", vecpref, buf, num2str(kk), ".mtx");
    mmwrite(matname,Dd);
    mmwrite(vecname,rs);
    matname = strcat("block_J_" , matpref, buf, num2str(kk), ".mtx");
    vecname = strcat("block_ry_", vecpref    , buf, num2str(kk), ".mtx");
    mmwrite(matname,Jc1);
    mmwrite(vecname,ry);
    matname = strcat("block_Jd_" , matpref, buf, num2str(kk), ".mtx");
    vecname = strcat("block_ryd_", vecpref    , buf, num2str(kk), ".mtx");
    mmwrite(matname,Jd);
    mmwrite(vecname,ryd);
end