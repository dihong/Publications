function X=L2Norm(X)
X=bsxfun(@rdivide,X,sqrt(sum(X.^2,1)));
end