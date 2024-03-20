 
clear, clc
load BCdat_error_bar
nB=1:10;
mu_val=[mean(r1000(:,2)),mean(r2000(:,2)),mean(r3000(:,2)),mean(r4000(:,2)),mean(r5000(:,2)),...
    mean(r10000(:,2)),mean(r20000(:,2)),mean(r30000(:,2)),mean(r50000(:,2)),mean(r100000(:,2))];
std_val=[std(r1000(:,2)),std(r2000(:,2)),std(r3000(:,2)),std(r4000(:,2)),std(r5000(:,2)),...
    std(r10000(:,2)),std(r20000(:,2)),std(r30000(:,2)),std(r50000(:,2)),std(r100000(:,2))];
errorbar(nB,mu_val,std_val)