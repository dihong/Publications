function distances = prmi_compute_pairwise_distance(X,Y,d)
% PRMI_COMPUTE_PAIRWISE_DISTANCE computes the euclidean pairwise distances.
%
%   distances = PRMI_COMPUTE_PAIRWISE_DISTANCE(X):     Euclidean distance between (X,X)
%      'X'         is DxP matrix, one column per feature.
%      'distances' is PxP matrix, where distances(p,q) = norm(X(:,p)-X(:,q)).
%
%   distances = PRMI_COMPUTE_PAIRWISE_DISTANCE(X,Y):   Euclidean distance between (X,Y)
%      'X'         is DxP matrix, one column per feature.
%      'Y'         is DxQ matrix, one column per feature.
%      'distances' is PxQ matrix, where distances(p,q) = norm(X(:,p)-Y(:,q)).
%
%   distances = PRMI_COMPUTE_PAIRWISE_DISTANCE(X,Y,d): Euclidean distance
%   between (X,Y), weighted by d vector.
%      'X'         is DxP matrix, one column per feature.
%      'Y'         is DxQ matrix, one column per feature.
%      'd'         is Dx1 vector, all elements must be non-negative.
%      'distances' is PxQ matrix, where distances(p,q) = norm((X(:,p)-Y(:,q)).*d).
%
% See also PRMI_L2_NORM
%
% Contact: www.dihong.me
if nargin == 1
    dx = sum(X.*X,1);
    distances = sqrt(bsxfun(@plus,dx',dx) - 2*(X'*X));
elseif nargin==2
    dx = sum(X.*X,1);
    dy = sum(Y.*Y,1);
    distances = sqrt(bsxfun(@plus,dx',dy) - 2*X'*Y);
elseif nargin==3
    if(any(d<0))
        error('All elements of d must be positive');
    elseif size(d,2) ~= 1
        error('The d must be column vector');
    end
    dx = sum(bsxfun(times,X.*X,d),1);
    dy = sum(bsxfun(times,Y.*Y,d),1);
    distances = sqrt(bsxfun(@plus,dx',dy) - 2*X'*diag(d)*Y);
end
end
