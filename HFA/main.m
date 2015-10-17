%
%% This is the main entry of the program.
% #Inputs: [based on Morph Album 2 dataset ONLY]
%       path1 - the path of the 10000 training persons of younger ages.
%       path2 - the path of the 10000 training persons of older ages.
%       path3 - the path of the 10000 testing persons of younger ages.
%       path4 - the path of the 10000 testing persons of older ages.

% #Output: rank_1 is the rank-1 recognition rate.

% #Comments: 'CoreNum' requires to be no more than the maximum MATLAB workers
% available in your local system. The file names of the face images must be
% the same as officially published names of Morph Album 2 dataset as this
% program extracts age information based on the file names.

% Auothr: Dihong Gong

% ============================Copyright=================================
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 1. Redistributions of source code must retain the above copyright
%    notice, this list of conditions and the following disclaimer.
% 2. Redistributions in binary form must reproduce the above copyright
%    notice, this list of conditions and the following disclaimer in the
%    documentation and/or other materials provided with the
%    distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
% "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
% A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
% HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
% SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
% LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
% DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
% THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function rank_1 = main(path1,path2,path3,path4)
CoreNum  = 12;
MATLABPOOL=1;
if ~MATLABPOOL && (matlabpool('size')>0)
    matlabpool close;
end
if MATLABPOOL
    if(matlabpool('size')~=CoreNum && matlabpool('size')>0)
        matlabpool close;
        matlabpool('open',CoreNum) %open local.
    elseif(matlabpool('size')<1)
        matlabpool('open',CoreNum) %open local.
    end
end
mex HOG.cpp
%% settings.
% path1 = 'train1'; %path of train-young
% path2 = 'train2'; %path of train-old
% path3 = 'test1'; %path of test-young
% path4 = 'test2'; %path of test-old
fls1 = dir([path1 '\*.bmp']);
fls2 = dir([path2 '\*.bmp']);
fls3 = dir([path3 '\*.bmp']);
fls4 = dir([path4 '\*.bmp']);
scale = [1 .75 .5];
szpatch = 48;
sift = 0.15;
nslice = 6;
% First level: USA to reduce feature dimension using the first nt1 training persons.
fprintf('training-1 feature extraction.\n');
[f1 f2] = feature_extraction(path1,path2,fls1,fls2,scale,szpatch,sift,1:5000);

%slice data:
[D N] = size(f1);
d = round(D/nslice);
for i = 1:nslice-1
    train1{i} = f1((i-1)*d+1:i*d,:);
    train2{i} = f2((i-1)*d+1:i*d,:);
end
train1{nslice} = f1((nslice-1)*d+1:end,:);
train2{nslice} = f2((nslice-1)*d+1:end,:);
clear f1 f2
%USA:
d3=1000;
fprintf('First level USA.\n');
parfor i = 1:nslice
    [PM{i} M{i}] = USA(train1{i},train2{i},d3+1,d3+1,d3);
end
clear train1 train2 functions

%training feature projection.
fprintf('training-2 feature extraction.\n');
[f1 f2] = feature_extraction(path1,path2,fls1,fls2,scale,szpatch,sift,5001:10000);
for i = 1:nslice-1
    train1{i} = f1((i-1)*d+1:i*d,:);
    train2{i} = f2((i-1)*d+1:i*d,:);
end
train1{nslice} = f1((nslice-1)*d+1:end,:);
train2{nslice} = f2((nslice-1)*d+1:end,:);
clear f1 f2;
for i = 1:nslice
    train1{i} = PM{i}'*bsxfun(@minus,train1{i},M{i});
    train2{i} = PM{i}'*bsxfun(@minus,train2{i},M{i});
end
clear functions
%testing feature projection.
fprintf('testing feature extraction.\n');
[f3 f4] = feature_extraction(path3,path4,fls3,fls4,scale,szpatch,sift,1:10000);

for i = 1:nslice-1
    test1{i} = f3((i-1)*d+1:i*d,1:end);
    test2{i} = f4((i-1)*d+1:i*d,1:end);
end
test1{nslice} = f3((nslice-1)*d+1:end,1:end);
test2{nslice} = f4((nslice-1)*d+1:end,1:end);
clear f3 f4;
for i = 1:nslice
    test1{i} = PM{i}'*bsxfun(@minus,test1{i},M{i});
    test2{i} = PM{i}'*bsxfun(@minus,test2{i},M{i});
end
%% second level: HFA
fprintf('Second level HFA.\n');
K = 8; %eight groups.
group = grouping_on_ages(path1,path2,K);
group = group(:,5001:10000);
parfor i = 1:nslice
    feat = generate_grouped_feat(group,L2Norm(train1{i}),L2Norm(train2{i}));
    [beta{i} U{i} V{i} sigma{i}] = train_model(feat);
end
rst = 0;
for i = 1:nslice
    gallery = compute_identity_factor(L2Norm(test1{i}),beta{i},U{i},V{i},sigma{i});
    probe = compute_identity_factor(L2Norm(test2{i}),beta{i},U{i},V{i},sigma{i});
    rst = rst + L2Norm(probe)'*L2Norm(gallery);
end
rank_1 = computeTOP(rst,1);
end

function [beta U V sigma] = train_model(train_feat)
% configurations.
p = 100; %dimension of hidden identity factor (x).
q = 3; %dimension of hidden age factor (y).
d = size(cat(2,train_feat{:,1}),1); %dimension of observation (t).
Id = eye(d);
Ip = eye(p);
Iq = eye(q);
[K N] = size(train_feat); %number of training subjects.
%% model training
maxEpoch = 10;
T1 = zeros(d,N); % mean grouped by subject.
T2 = zeros(d,K); % mean grouped by ages.
T3 = cat(2,train_feat{:}); % all the training feat.
Ni = zeros(1,N); % # samples of the i-th subject.
Mk = zeros(1,K); % # samples of the k-th group.
beta = mean(T3,2);
T3 = bsxfun(@minus,T3,beta);
for i = 1:N
    tmp = cat(2,train_feat{:,i});
    T1(:,i) = mean(meanNorm(tmp,beta),2);
    Ni(i) = size(tmp,2);
end
for k = 1:K
    tmp = cat(2,train_feat{k,:});
    T2(:,k) = mean(meanNorm(tmp,beta),2);
    Mk(k) = size(tmp,2);
end
G = zeros(K,N);
for k = 1:K
    for n = 1:N
        G(k,n) = size(train_feat{k,n},2);
    end
end

% model parameters initialization.
rng('shuffle');
sigma = 0.1;
U = .2*(rand(d,p)-.5);
V = .2*(rand(d,q)-.5);

for it = 1:maxEpoch
    %compute first moment of x and y: Ex, Ey.
    S = sigma*Id+U*U'+V*V';
    Ex = U'/S*T1;
    Ey = V'/S*T2;
    %compute A,B,C,D,E,F.
    A = (Ip - U'/S*U)*N+Ex*sparse(diag(Ni))*Ex';
    B = (Iq - V'/S*V)*K+Ey*sparse(diag(Mk))*Ey';
    C = bsxfun(@times,T1,Ni)*Ex';
    D = bsxfun(@times,T2,Mk)*Ey';
    E = -V'/S*U*sum(sum(1./(sqrt(Mk)*G*sqrt(Ni)')))+Ey*G*Ex';
    F = E';
    Ux = U*Ex;
    Vy = V*Ey;
    %update U, V.
    U = (C-D/B*E)/(A-F/B*E);
    V = (D-C/A*F)/(B-E/A*F);
    %update sigma.
    T4 = cell(K,N);
    for k = 1:K
        for n = 1:N
            if size(train_feat{k,n},2) == 1
                T4{k,n} = Ux(:,n)+Vy(:,k);
            elseif size(train_feat{k,n},2) == 2
                T4{k,n} = repmat(Ux(:,n)+Vy(:,k),1,2);
            else
                T4{k,n} = [];
            end
        end
    end
    sigma = sum(sum((T3-cat(2,T4{:})).*T3))/numel(T3);
end
end


function Ex = compute_identity_factor(test_feat,beta,U,V,sigma)
d = length(beta);
S = sigma*eye(d)+U*U'+V*V';
Ex = U'/S*bsxfun(@minus,test_feat,beta);
end

function feat = generate_grouped_feat(group,train1,train2)
feat = cell(size(group));
for i = 1:size(group,1)
    for j = 1:size(group,2)
        if group(i,j) == 1
            feat{i,j} = [feat{i,j} train1(:,j)];
        elseif group(i,j) == 2
            feat{i,j} = [feat{i,j} train2(:,j)];
        elseif group(i,j) == 3
            feat{i,j} = [feat{i,j} train1(:,j) train2(:,j)];
        end
    end
end
end

function [F1 F2] = feature_extraction(path1,path2,fls1,fls2,scale,szpatch,sift,index)
F1 = []; F2 = [];
for s = scale
    szImage=size(imresize(imread([path1 '\' fls1(1).name]),s));
    parfor i = index
        f1{i}=HOG(double(imresize(imread([path1 '\' fls1(i).name ]),s)),szpatch,sift,szImage);
        f2{i}=HOG(double(imresize(imread([path2 '\' fls2(i).name ]),s)),szpatch,sift,szImage);
    end
    clear functions
    f1 = cat(2,f1{:});
    f2 = cat(2,f2{:});
    F1 = [F1;f1];
    F2 = [F2;f2];
    clear f1 f2;
end
end

function [Y m]=meanNorm(X,meanVec)
if nargin==1
    m=mean(X,2);
else
    m=meanVec;
end
Y=bsxfun(@minus,X,m);
end

function X=L2Norm(X)
L2 = sqrt(sum(X.^2,1));
X=bsxfun(@rdivide,X,L2);
if(any(L2==0))
    X(:,L2==0) = zeros(size(X,1),sum(L2==0));
end
end

function avgTop=computeTOP(rst,top)
if nargin == 1 
    top = 1;
end
score=zeros(top,size(rst,1));
for i = 1:size(rst,1)
    [temp index] = sort(rst(i,:),'ascend');
    for k2 = 1:top
        score(k2,i) = score(k2,i) + ~isempty(find(index((end-k2+1):end) ==i,1));
    end
end
avgTop=sum(score,2)/size(score,2);
end
