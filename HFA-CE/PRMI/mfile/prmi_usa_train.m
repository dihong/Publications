function model = prmi_usa_train(data,label,d1,d2)
% PRMI_USA_TRAIN Universal Subspace Analysis.
% model = PRMI_USA_TRAIN(data,label,d1,d2) performs USA training:
%   'data'    A DxN training data matrix, one sample per column.
%   'label'   A vector indicating the labels for training samples.
%   'd1'      An integer for dimension of within-class whitening(d1<min(size(data))).
%   'd2'      An integer for dimension of between-class whitening (d2<=d1).
%   'model'   The trained USA model for classification.
%
% See also PRMI_USA_TEST, PRMI_MCA_TRAIN.
%
% Contact: www.dihong.me

if d1 > min(size(data))
    error(['USA: d1 must be smaller or equal to ' num2str(min(size(data))) '.']);
elseif d2 > d1
    error('USA: d2 must be smaller or equal to d1.');
elseif size(data,2) ~= numel(label)
    error('USA: label must be of the same size as number of training samples.');
end

%run the USA.
c = unique(label);
nclass = numel(c);
M = mean(data,2);
classcenters = zeros(size(data,1),nclass);
for i = 1:nclass
    classcenters(:,i) = mean(data(:,label==c(i)),2);
    data(:,label==c(i)) = bsxfun(@minus,data(:,label==c(i)),classcenters(:,i));
end
[W1, ~, eigenvalues]=prmi_pca(data,'ndim',d1);
W1 = W1(:,eigenvalues>1e-10);
eigenvalues = eigenvalues(eigenvalues>1e-10);
W1 = W1/sqrt(diag(eigenvalues));
classcenters = bsxfun(@minus,classcenters,mean(classcenters,2));
W2 = prmi_pca(W1'*classcenters,'ndim',d2);
P = W1*W2;
model.P = P;
model.M = M;
end
