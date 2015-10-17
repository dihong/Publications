%% This function compute MAE of FG-NET dataset based on Automatic face detection.
function fgnet
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
load('fgnet_faces','faces','age','id');
age4 = cat(2,age{:});
id4 = cat(2,id{:});

feat = [log(1+0.1*extract_feat(faces,'dsift'));log(1+0.1*extract_feat(faces,'mlbp'))];%
age = age4;
id = id4;

[P M] = PCA(feat,0.99);
feat = P'*bsxfun(@minus,feat,M);

uid = unique(id);
tic
parfor tid = 1:length(uid)
    mae = mtwgp_regression(feat,age,id,uid(tid));
    E{tid} = mae;
end
E = cat(2,E{:});
mean(E)
%             fprintf('sigma = %.2f, mu = %.2f, rho = %.2f, accuracy = %.4f\n',sigma,mu,rho,mean(E));
clear E;
toc
end


% uy = unique(y);
% for i = 1:length(uy)
%     e(i) = mean(abs(m(y==uy(i))-y(y==uy(i))));
% end

