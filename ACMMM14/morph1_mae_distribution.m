%% This function compute MAE of FG-NET dataset based on Automatic face detection.
function morph1_mae_distribution
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
load('morph1_faces','faces','age','id');
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
    [mae test{tid} pred{tid}] = mtwgp_regression(feat,age,id,uid(tid));
end
test = cat(2,test{:});
pred = cat(2,pred{:});

uy = unique(test);
for i = 1:length(uy)
    e(i) = mean(abs(pred(test==uy(i))-test(test==uy(i))));
end

plot(uy,e,'r.-');
xlabel('Age (year)');
ylabel('Mean Age Error (year)');
%             fprintf('sigma = %.2f, mu = %.2f, rho = %.2f, accuracy = %.4f\n',sigma,mu,rho,mean(E));
clear E;
toc
end




