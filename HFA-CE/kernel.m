function K = kernel(X,Y,type,p1,p2)
if strcmp(type,'Linear')
    K = X'*Y;
elseif strcmp(type,'Gaussian')
    D = bsxfun(@plus,sum(X.*X,1)',sum(Y.*Y,1))-2*X'*Y;
    K = exp(-D/(2*p1*p1));
elseif strcmp(type,'Sigmoid')
    K = tanh(p1*X'*Y+p2);
else
    error('error using kernel: undefined type');
end
end