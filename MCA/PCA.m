function [eigenvectors, meanVector, eigenvalues]=PCA(X,N)
[Row Column]=size(X);
%Mean center X
meanVector = mean(X,2);
X = bsxfun(@minus,X,meanVector);

if size(X,1) > size(X,2)
    C=X'*X./Column+eye(Column)*1e-6;
    [V,D]=eig(C);
    eigenvalues=diag(D);
    
    %Ordered by eigenvalues%
    [eigenvalues,Index]=sort(eigenvalues,'descend');
    
    V=V(:,Index) ;   %V1 is the the eigenvectors got from X'X;
    
    eigenvectors=X*V;%eigenvectors is the eigenvectors for XX';
    
    %L2-normalization.
    NV=sum(eigenvectors.^2,1);
    NV=NV.^(1/2);
    NM=repmat(NV,Row,1);
    eigenvectors=eigenvectors./NM;
else
    C=X*X'./Column+eye(Row)*1e-6;
    [V,D]=eig(C);
    eigenvalues=diag(D);
    [eigenvalues,Index]=sort(eigenvalues,'descend');
    eigenvectors=V(:,Index) ;   %V1 is the the eigenvectors got from X'X;
end

if nargin>1
    if N < 1
        p = N*sum(eigenvalues);
        e = 0;
        for i = 1:length(eigenvalues)
            e = e + eigenvalues(i);
            if e>=p
                break;
            end
        end
        N = i;
    end
    eigenvectors=eigenvectors(:,1:N);
    eigenvalues = eigenvalues(1:N);
end

end