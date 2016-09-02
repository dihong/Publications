function [P,M]=prmi_usa(X,label,d1,d2)
% PRMI_USA Universal Subspace Analysis.
% [P,M]=prmi_usa(X,label,d1,d2) performs USA on data matrix X whose columns 
% consist of observations. To project a test sample t into USA subspace: 
% tp = P'*(t-M).
%
%   'label' is a vector indicating the labels for training samples X.
%   'd1' is an integer for dimension of within-class whitening(d1<min(size(X))).
%   'd2' is an integer for dimension of between-class whitening (d2<=d1).
%
% See also PRMI_MCA_TRAIN, PRMI_USA_TRAIN, PRMI_PCA
%
% Contact: www.dihong.me

X = double(X);

if d1 > min(size(X))
    error(['USA: d1 must be smaller or equal to ' num2str(min(size(X))) '.']);
elseif d2 > d1
    error('USA: d2 must be smaller or equal to d1.');
elseif size(X,2) ~= numel(label)
    error('USA: label must be of the same size as number of training samples.');
end

%run the USA.
c = unique(label);
nclass = numel(c);
M = mean(X,2);
classcenters = zeros(size(X,1),nclass);
for i = 1:nclass
    classcenters(:,i) = mean(X(:,label==c(i)),2);
    X(:,label==c(i)) = bsxfun(@minus,X(:,label==c(i)),classcenters(:,i));
end
[W1, ~, eigenvalues]=prmi_pca(X,'ndim',d1);
W1 = W1(:,eigenvalues>1e-10);
eigenvalues = eigenvalues(eigenvalues>1e-10);
W1 = W1/sqrt(diag(eigenvalues));
classcenters = bsxfun(@minus,classcenters,mean(classcenters,2));
W2 = prmi_pca(W1'*classcenters,'ndim',d2);
P = W1*W2;
end
