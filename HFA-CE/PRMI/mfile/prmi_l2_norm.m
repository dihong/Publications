function X = prmi_l2_norm(X)
% PRMI_L2_NORM performs L2 normalization.
% X = PRMI_L2_NORM(X) computes the L2 normalized version of X. The
% normalization is performed on columns of X.
%
% Contact: www.dihong.me
L2 = sqrt(sum(X.^2,1));
X=bsxfun(@rdivide,X,L2);
if(any(L2==0))
    X(:,L2==0) = zeros(size(X,1),sum(L2==0));
end
end