function main
CoreNum = 6;
MATLABPOOL = 1;
profile = 'local';
poolobj = gcp('nocreate'); % If no pool, do not create new one.
if isempty(poolobj)
    poolsize = 0;
else
    poolsize = poolobj.NumWorkers;
end
if ~MATLABPOOL && poolsize>0
    delete(poolobj);
end
if MATLABPOOL
    if(poolsize~=CoreNum && poolsize>0)
        delete(poolobj);
        parpool(profile,CoreNum);
    elseif(poolsize<1)
        parpool(profile,CoreNum);
    end
end
run('Tools/vlfeat-0.9.17-bin/vlfeat-0.9.17/toolbox/vl_setup');

%% Feature Extraction.
train1 = 'train/1'; %train path 1
train2 = 'train/2'; %train path 2
test1 = 'test/1'; %test path 1
test2 = 'test/2'; %test path 2
Train1 = feature_extraction(train1,'bmp','dsift');
Train2 = feature_extraction(train2,'bmp','dsift');
Test1 = feature_extraction(test1,'bmp','dsift');
Test2 = feature_extraction(test2,'bmp','dsift');

% tmp = Test1;
% Test1(:,10001:end)=[];
% Test1 = [tmp(:,10001:end) Test1];

nslice = 18;
[D N] = size(Train1);
d = round(D/nslice);
clear train1 train2 test1 test2
for i = 1:nslice-1
    train1{i} = Train1((i-1)*d+1:i*d,:);
    train2{i} = Train2((i-1)*d+1:i*d,:);
    test1{i} = Test1((i-1)*d+1:i*d,:);
    test2{i} = Test2((i-1)*d+1:i*d,:);
end
train1{nslice} = Train1((nslice-1)*d+1:end,:);
train2{nslice} = Train2((nslice-1)*d+1:end,:);
test1{nslice} = Test1((nslice-1)*d+1:end,:);
test2{nslice} = Test2((nslice-1)*d+1:end,:);

rst = 0;
tic
parfor i = 1:nslice
    [P M beta U sigma] = MCA({train1{i},train2{i}},400,250);
    [u1 C1]=computeEy(test1{i},P,M,beta{1},U{1},sigma{1});
    [u2 C2]=computeEy(test2{i},P,M,beta{2},U{2},sigma{2});
    rst = rst + MDist(u1,u2,C1,C2);
end
toc
%% Visualization.
top = computeTOP(rst,1)
end

function rst = MDist(g,p,Cg,Cp)
g = L2Norm(g);
p = L2Norm(p);
invCg = inv(Cg);
invCp = inv(Cp);
invA = inv(invCg + invCp);
Vg = invCg*g;
W = invCp*invA*Vg;
W0 = diag(Vg'*invA*Vg - g'*invCg*g)'/2; %row vector.
rst = bsxfun(@plus,W0,p'*W);
end