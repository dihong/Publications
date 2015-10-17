%% nargin = 1: squared Euclidean distance between (X,X)
%  nargin = 2: squared Euclidean distance between (X,Y)
%  nargin = 3: squared Euclidean distance between (X,Y) weighted by d (column vector):
%  D(i,j) = sum_k(X[i,k]*Y[j,k]*d[k])
function [D C] = compute_pairwise_distance(X,Y,d)
if nargin == 1
    dx = sum(X.*X,1);
    C = X'*X;
    D = bsxfun(@plus,dx',dx) - 2*C;
elseif nargin==2
    dx = sum(X.*X,1);
    dy = sum(Y.*Y,1);
    C = X'*Y;
    D = bsxfun(@plus,dx',dy) - 2*C;
elseif nargin==3
    dx = sum(bsxfun(@times,X.*X,d),1);
    dy = sum(bsxfun(@times,Y.*Y,d),1);
    C = X'*diag(d)*Y;
    D = bsxfun(@plus,dx',dy) - 2*C;
end
end