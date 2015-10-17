%% This function compute the MCA subspace.
% [t] a dx(2n) matrix consist of 2n samples from n subjects (each subject has 2 samples from alternative modality). The
% training observations should be arranged as: [t11 t12 t21 t22 t31
% t32...tn1 tn2].
% [dh] is the dimension of latent variables.
% [rank_u] is the rank of the matrix U.
% return: [model] is a model for classification.

function [P M beta U sigma] =MCA(train,dpca,dim)

%PCA.

[P M] = PCA([train{1} train{2}],dpca);
train{1} = P'*bsxfun(@minus,train{1},M);
train{2} = P'*bsxfun(@minus,train{2},M);

%MCA.

beta{1} = mean(train{1},2);
beta{2} = mean(train{2},2);

train{1} = bsxfun(@minus,train{1},beta{1});
train{2} = bsxfun(@minus,train{2},beta{2});

MaxEpoch = 10;

[neigenvectors, meanVector,eigVal] = PCA((train{1}+train{2})/2,dim);

[D N] = size(train{1});

U{1} = neigenvectors*diag(sqrt(eigVal));%(rand(D,dim)-.5)*.2;%
U{2} = U{1};
sigma{1} = 0.1;
sigma{2} = sigma{1};

I = eye(dim);

I2 = eye(D);

for it = 1:MaxEpoch
    % E-step
    Ex = 0; Exx = 0;
    for i = 1:2
        UC = U{i}'/(sigma{i}*I2+U{i}*U{i}');
        tmp = UC*train{i};
        Ex = Ex + tmp;
        Exx = Exx + (I-UC*U{i})*N + tmp*tmp';
    end
    
    Ex = Ex/2;
    Exx = Exx/2;
    
    % M-step
    for i = 1:2
        U{i} = train{i}*Ex'/Exx;
        sigma{i} = sum(sum((train{i}-U{i}*Ex).*train{i}))/(N*D)/2;
    end
    
    err(it) = computePerf(train,U,Ex,beta);
    
    if it>3 && err(end)>(1-1e-2)*err(end-2)
        break;
    end
    
end

end


function perf = computePerf(train,U,Ex,beta)
perf=0;
for k=1:2
    m=bsxfun(@plus,U{k}*Ex,beta{k});
    DM=(train{k}-m).^2;
    perf=perf+sqrt(sum(DM(:)));
end
end
