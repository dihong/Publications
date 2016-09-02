%% This is an entry to the testing program for PRMI.
% It is based on FERET dataset.
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
run('../../prmi_setup.m');
%% load faces.
% image1 = cell(1,1194);
% image2 = cell(1,1194);
% for i = 1:1194
%         image1{i} = imread(['../feret/200x160/photo/' num2str(i) '.bmp']);
%         image2{i} = imread(['../feret/200x160/skecth/' num2str(i) '.bmp']);
% end
load('ferect','image1','image2');
% tic
% model =  prmi_lps_train(image1(1:500),image2(1:500),'scales',[1 3 5 7],'alpha',0.3,'ncode',64);
% save('model','model');
% toc
load('model','model');
%% extract features.
% prmi_lps(model,image1{1},'winsize',20,'step',20);
model = model;
tic
parfor i = 1:length(image1)
    feat1{i} = prmi_lps(model,image1{i},'winsize',20,'step',8);
    feat2{i} = prmi_lps(model,image2{i},'winsize',20,'step',8); %,'winsize',16,'step',8,'numbins',12
end
% parfor i = 1:length(image1)
%     feat1{i} = prmi_mlbp(image1{i},'winsize',20,'step',8);
%     feat2{i} = prmi_mlbp(image2{i},'winsize',20,'step',8); %,'winsize',16,'step',8,'numbins',12
% end
toc
feat1 = cat(2,feat1{:});
feat2 = cat(2,feat2{:});
nslice = 18;
dpca = 500;
d1 = 250;
d2 = 200;
label = [1:500 1:500];

%split data.
train1 = feat1(:,1:500);
train2 = feat2(:,1:500);
test1 = feat1(:,501:end);
test2 = feat2(:,501:end);
%slice data.
train1 = prmi_slice(train1,nslice);
train2 = prmi_slice(train2,nslice);
test1 = prmi_slice(test1,nslice);
test2 = prmi_slice(test2,nslice);
%% This is testing for prmi_random_forest
% PCA dimension reduction.
for j = 1:nslice
    [P,M] = prmi_pca((train1{j}+train2{j})/2,'ndims',dpca);
    train1{j} = P'*bsxfun(@minus,train1{j},M);
    train2{j} = P'*bsxfun(@minus,train2{j},M);
    test1{j} = P'*bsxfun(@minus,test1{j},M);
    test2{j} = P'*bsxfun(@minus,test2{j},M);
end
%random forest.
nbags = 1;
sa = 1;
sb = 1;
scores = 0;
tic
parfor j = 1:nslice
    model = prmi_random_forest_train([train1{j} train2{j}],label,@prmi_usa_train,{d1,d2},@prmi_usa_test,nbags,sa,sb);
    scores = scores + prmi_random_forest_test(model,test2{j},test1{j});
end
top = prmi_compute_top(scores)
toc
return;
%%
%train subspace.
tic
parfor j = 1:nslice
    [P1,M1] = prmi_pca((train1{j}+train2{j})/2,'ndim',dpca);
    train1{j} = P1'*bsxfun(@minus,train1{j},M1);
    train2{j} = P1'*bsxfun(@minus,train2{j},M1);
    P2 = prmi_usa([train1{j} train2{j}],label,d1,d2);
    P{j} = P1*P2;
    M{j} = M1;
end
toc
%project testing persons into subspace representation.
for j = 1:nslice
    test1{j} = P{j}'*bsxfun(@minus,test1{j},M{j});
    test2{j} = P{j}'*bsxfun(@minus,test2{j},M{j});
end
%face matching.
rst = 0;
for j = 1:nslice
    rst = rst + DistanceMatrix(test1{j}, test2{j},'cosine');
end
top = computeTOP(rst)
end