function X=L2Norm(X)
L2 = sqrt(sum(X.^2,1));
X=bsxfun(@rdivide,X,L2);
if(any(L2==0))
    X(:,L2==0) = zeros(size(X,1),sum(L2==0));
end
end