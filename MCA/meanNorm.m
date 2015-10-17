function [Y m]=meanNorm(X,meanVec)
if nargin==1
    m=mean(X,2);
else
    m=meanVec;
end
Y=bsxfun(@minus,X,m);
end