function [neigenvectors, meanVector,eigenvalues,eigenvectors]=PCA(X,N)
[Row Column]=size(X);
%Mean center X
[X meanVector]=meanNorm(X);
if size(X,1) > size(X,2)
    C=X'*X./Column;
    [V,D]=eig(C,'nobalance');
    eigenvalues=diag(D);
    
    %Ordered by eigenvalues%
    [eigenvalues,Index]=sort(eigenvalues,'descend');
    
    V=V(:,Index) ;   %V1 is the the eigenvectors got from X'X;
    
    eigenvectors=X*V;%eigenvectors is the eigenvectors for XX';
else
    C=X*X'./Column;
    [V,D]=eig(C,'nobalance');
    eigenvalues=diag(D);
    [eigenvalues,Index]=sort(eigenvalues,'descend');
    eigenvectors=V(:,Index) ;   %V1 is the the eigenvectors got from X'X;
end
%normalize%
NV=sqrt(sum(eigenvectors.^2));

%normalize eigenvectors
neigenvectors=bsxfun(@rdivide,eigenvectors,NV);

if nargin>1
    neigenvectors=neigenvectors(:,1:N);
    eigenvectors = eigenvectors(:,1:N);
    eigenvalues = eigenvalues(1:N);
end

if ~isreal(eigenvectors(:)) || ~isreal(neigenvectors(:))
    error('non-real');
end

end