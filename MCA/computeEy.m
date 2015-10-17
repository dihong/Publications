function [u C]=computeEy(data,P,M,beta,U,sigma)
% Estep function is to compute the first and second moment of P(y(s)|Mk(1,2...),V,M)
% compute the distribution parameters of P(y(s)|Mk(1,2...),V,M).
data = P'*bsxfun(@minus,data,M);
UC = U'/(sigma*eye(size(U,1))+U*U');
u = UC*bsxfun(@minus,data,beta);
C = (eye(size(U,2))-UC*U);
end