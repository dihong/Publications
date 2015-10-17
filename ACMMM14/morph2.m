%% This function compute MAE of MORPH-1 dataset based on Automatic face detection.
function morph2
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
run('../Tools/vlfeat-0.9.17-bin/vlfeat-0.9.17/toolbox/vl_setup');
load('morph2_young_faces','faces','age','id');
age1 = cat(2,age{:});
id1 = cat(2,id{:});
feat1 = extract_feat(faces,'dsift');

feat1 = [feat1;extract_feat(faces,'mlbp')];

load('morph2_old_faces','faces','age','id');
age2 = cat(2,age{:});
id2 = cat(2,id{:});
feat2 = extract_feat(faces,'dsift');
feat2 = [feat2;extract_feat(faces,'mlbp')];

clear faces;

age = [age1 age2];
id = [id1 id2];

feat = single([feat1 feat2]);

clear feat1 feat2;

% tic
% feat = log(1+0.1*feat);
% [P M] = PCA(feat,0.99);
% feat = P'*bsxfun(@minus,feat,M);
% toc
% save('morph_subspace','P','M');

uid = unique(id);

rind = randperm(numel(uid));

tid = uid(rind<=1000);

for i = 1:length(tid)
    id(id==tid(i)) = 0;
end

ntrain = sum(id~=0)

ntest = sum(id==0)

tic
mae = mtwgp_regression(feat,age,id,0)
toc
%             fprintf('sigma = %.2f, mu = %.2f, rho = %.2f, accuracy = %.4f\n',sigma,mu,rho,mean(E));
end

