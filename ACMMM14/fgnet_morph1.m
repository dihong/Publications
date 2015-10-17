%% This function compute MAE of FG-NET dataset based on Automatic face detection.
function fgnet_morph1
CoreNum=12;
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
% run('../Tools/vlfeat-0.9.17-bin/vlfeat-0.9.17/toolbox/vl_setup');
load('fgnet_faces','faces','age','id');
age1 = cat(2,age{:});
id1 = cat(2,id{:});
feat1 = extract_feat(faces,'dsift');
feat1 = [feat1;extract_feat(faces,'mlbp')];

load('morph1_faces','faces','age','id');
age2 = cat(2,age{:});
id2 = cat(2,id{:});
feat2 = extract_feat(faces,'dsift');
feat2 = [feat2;extract_feat(faces,'mlbp')];


feat = [feat1 feat2];
age = [age1 age2];
id = [id1 id2];

feat = log(1+0.1*feat);

[P M] = PCA(feat,0.99);
feat = P'*bsxfun(@minus,feat,M);

uid = unique(id);
uid2 = unique(id2);
rind = randperm(length(uid2));
tic
for i = 0
    i
    feat3 = feat(:,1:length(age1));
    age3 = age(1:length(age1));
    id3 = id(1:length(age1));
    for j = 1:i
        feat3 = [feat3 feat(:,id==uid(rind(j)))];
        age3 = [age3 age(:,id==uid(rind(j)))];
        id3 = [id3 id(:,id==uid(rind(j)))];
    end
    parfor tid = 1:82
        mae = mtwgp_regression(feat3,age3,id3,tid);
        E{tid} = mae;
    end
    E = cat(2,E{:});
    mean(E)
    clear E;
end
%             fprintf('sigma = %.2f, mu = %.2f, rho = %.2f, accuracy = %.4f\n',sigma,mu,rho,mean(E));
clear E;
toc
end