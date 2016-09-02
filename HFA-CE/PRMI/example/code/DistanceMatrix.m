function rst=DistanceMatrix(G,P,distance,map)
if nargin<3
    distance='cosine';
end
% if strcmp(distance,'cosine')
%     G=L2Norm(G);
%     P=L2Norm(P);
% end
pCombined=L2Norm(P);
P=[];
gCombined=L2Norm(G);
G=[];
sz2P=size(P,2);
sz2G=size(G,2);
rstCombined=zeros(sz2P,sz2G);
if strcmp(distance,'cosine')
    rstCombined=pCombined'*gCombined;%Prodpg(i,j) is the product for vectors of sketch(i) and photo(j).
elseif strcmp(distance,'euclid');
    if matlabpool('size')>0
        CoreNum=matlabpool('size');
        szSlice=floor(sz2P/CoreNum);
        p=cell(CoreNum,1);
        rstCombined=cell(CoreNum,1);
        for i=1:CoreNum-1
            p{i}=pCombined(:,(i-1)*szSlice+1:i*szSlice);
        end
        p{CoreNum}=pCombined(:,(CoreNum-1)*szSlice+1:end);
        parfor i=1:CoreNum
            rstCombined{i}=feuclid(p{i},gCombined);
        end
        rstCombined=-cat(1,rstCombined{:});
    else
        for i=1:sz2P
            rstCombined(i,:)=-sum((repmat(pCombined(:,i),1,sz2G)-gCombined).^2,1);
        end
    end
else
    error('unkown distance type');
end

rst=(rstCombined);
end

%% function for Euclid distance computation.
function rst=feuclid(p,g,C)
rst=zeros(size(p,2),size(g,2));
for j=1:size(p,2);
    rst(j,:)=sum(bsxfun(@minus,p(:,j),g).^2,1);
end
end