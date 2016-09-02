function [eigenvectors, meanvector, eigenvalues] = prmi_pca(varargin)
% PRMI_PCA Principal Components Analysis.
% [eigenvectors, meanvector, eigenvalues]=prmi_pca(X) performs PCA on data
% matrix X whose columns consist of observations. The columns of eigenvectors
% are sorted in descending order of the corresponding eigenvalues. To
% project a test sample t into PCA subspace: tp = P'*(t-meanvector).
%
% [eigenvectors, meanvector, eigenvalues] = prmi_pca(X,OPTION1,OPTION2) performs PCA with specified options.
% Options arguments can have the following values:
%
%   'ratio'   The ratio is a value between 0 and 1 which specifies the ratio 
%             of energy to be preserved. When the 'ndims' option is
%             specified, the 'ratio' option will be ignored. By default,
%             ratio = 1.
%   'ndims'    The ndims is an integer between 1 and K where K = min(size(X))
%             that specifies the number of eigenvectors to be preserved. By
%             default, ndims = K.
%
% Contact: www.dihong.me

if length(varargin)<1
    error('PCA: At least one argument required.');
else
    X = varargin{1};
end

validateattributes(X,{'double'},{'nonsparse','finite','real','nonnan','2d'},mfilename,'X',1);

allStrings = {'ratio', 'ndims'};

%defaults
K = min(size(X));
ratio = 1;
ndims = K;

%parse inputs.
for k = 2:2:length(varargin)
  if ischar(varargin{k})
    string = lower(validatestring(varargin{k}, allStrings, mfilename, 'OPTION',k));
    switch string
        case {'ratio'}
            validateattributes(varargin{k+1},{'double'}, {'real','>',0,'<=',1},mfilename,'ratio',1+k);
            ratio = varargin{k+1};
        case {'ndims'}
            validateattributes(varargin{k+1},{'double'}, {'integer','>',0,'<=',K},mfilename,'ndims',1+k);
            ndims = varargin{k+1};
    end
  else
      error('PCA: incorrect option format.');
  end
end

%run the PCA.
[Row,Column]=size(X);
%Mean center X
meanvector = mean(X,2);
X = bsxfun(@minus,X,meanvector);
if size(X,1) > size(X,2)
    C=X'*X./Column;
    [V,D]=eig(C);
    eigenvalues=diag(D);
    %Ordered by eigenvalues
    [eigenvalues,Index]=sort(eigenvalues,'descend');
    V=V(:,Index) ;   %V1 is the the eigenvectors got from X'X;
    eigenvectors=X*V;%eigenvectors is the eigenvectors for XX';
    %L2-normalization.
    NV=sum(eigenvectors.^2,1);
    NV=NV.^(1/2);
    NM=repmat(NV,Row,1);
    eigenvectors=eigenvectors./NM;
else
    C=X*X'./Column;
    [V,D]=eig(C);
    eigenvalues=diag(D);
    [eigenvalues,Index]=sort(eigenvalues,'descend');
    eigenvectors=V(:,Index) ;   %V1 is the the eigenvectors got from X'X;
end
%select some leading eigen vectors.
if ratio ~= 1 || ndims ~= K
    if ndims == K % select top eigvenvectors so that the corresponding eigenvalues preserve 100*T percent of energy.
        p = ratio*sum(eigenvalues);
        e = 0;
        for i = 1:length(eigenvalues)
            e = e + eigenvalues(i);
            if e>=p
                break;
            end
        end
        T = i;
    else
        T = ndims;
    end
    eigenvectors=eigenvectors(:,1:T);
    eigenvalues = eigenvalues(1:T);
end
eigenvalues(eigenvalues<=eps) = 0;
validateattributes(eigenvalues,{'double'},{'finite','nonnan','real'},mfilename,'eigenvalues');
validateattributes(eigenvectors,{'double'},{'finite','nonnan','real'},mfilename,'eigenvectors');
end