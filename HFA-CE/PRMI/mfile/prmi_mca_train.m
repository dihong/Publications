function model = prmi_mca_train(data,label,dim)
% PRMI_MCA_TRAIN Mutual Components Analysis.
% model = PRMI_MCA_TRAIN(data,label,dim) performs MCA on data matrix whose 
% columns consist of observations. The MCA is used for cross-modality
% subspace analysis. 
%
%   'data'    A DxN training data matrix, one sample per column.
%   'label'   A vector indicating the labels for training samples. The MCA
%             requires that there are exactly two samples for each class.
%   'dim'     The dimension of MCA subspace. The 'dim' must be less than or
%             equal to min([D N]). 
%
% See also PRMI_MCA_TEST, PRMI_USA_TRAIN, PRMI_PCA.
%
% Contact: www.dihong.me

ul = unique(label);
temp = data;
for i = 1:length(ul)
    if(sum(label==ul(i))) ~=2
        error('The number of samples for all the classes must be 2.');
    else
        temp(:,(i-1)*2+1:i*2) = data(:,label==ul(i));
    end
end
train{1} = temp(:,1:2:end);
train{2} = temp(:,2:2:end);

beta{1} = mean(train{1},2);
beta{2} = mean(train{2},2);

train{1} = bsxfun(@minus,train{1},beta{1});
train{2} = bsxfun(@minus,train{2},beta{2});

MaxEpoch = 10;

[neigenvectors, ~,eigVal] = prmi_pca((train{1}+train{2})/2,'ndims',dim);

[D,N] = size(train{1});

U{1} = neigenvectors*diag(sqrt(eigVal));
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
    
end

model.sigma = sigma;
model.beta = beta;
model.U = U;

end