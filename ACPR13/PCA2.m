function [eigenvalues, eigenvectors, Mean_Vector]=PCA2(X,N)


%Compute the eienvectors of X when the sample number is larger than the feature lenght


[Row Column]=size(X);
Mean_Vector=mean(X,2);
m=repmat(Mean_Vector,1,Column);
X=X-m;
C=X*X'./Column;
[V,D]=eig(C,'nobalance');
eigenvalues=diag(D);
%Ordered by eigenvalues%
[eigenvalues,Index]=sort(eigenvalues);
eigenvalues=eigenvalues(end:-1:1);
Index=Index(end:-1:1);
eigenvectors=V(:,Index);
if nargin>1
    eigenvectors=eigenvectors(:,1:N);
    eigenvalues = eigenvalues(1:N);
end

