%% This function performs PCA+USA: without L2-norm.
% data is DxN matrix with each column corresponding to one instance
% data is organized as: [g1,g2,...,gn,p1,p2,...,pn]
% d1,d2,d3 is the PCA dimension, USA dimensioins.
function [P M]=USA(g,p,d1,d2,d3)
%perform PCA.
if d1>0
    [Wpca M]=PCA([g p],d1);
    g = Wpca'*bsxfun(@minus,g,M);
    p = Wpca'*bsxfun(@minus,p,M);
else
    M = mean([g p],2);
    Wpca = eye(size(g,1));
end
%within-class whitening.
classcenters = (g+p)/2;
g = g - classcenters;
p = p - classcenters;
[W1, meanVector, eigenvalues]=PCA([g p],d2);
W1 = W1/sparse(sqrt(diag(eigenvalues)));
%between-class whitening.
classcenters = meanNorm(classcenters);
W2 = PCA(W1'*classcenters,d3);
P = Wpca*W1*W2;
end

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
    eigenvectors=eigenvectors(:,1:N);
    eigenvalues = eigenvalues(1:N);
end

end

function [Y m]=meanNorm(X,meanVec)
if nargin==1
    m=mean(X,2);
else
    m=meanVec;
end
Y=bsxfun(@minus,X,m);
end