function avgTop=computeTOP(rst,top)
if nargin == 1 
    top = 1;
end
score=zeros(top,size(rst,1));
for i = 1:size(rst,1)
    [temp index] = sort(rst(i,:),'ascend');
    for k2 = 1:top
        score(k2,i) = score(k2,i) + ~isempty(find(index((end-k2+1):end) ==i,1));
    end
end
avgTop=sum(score,2)/size(score,2);
end