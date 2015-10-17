function avgTop = computeTOP(rst,top)
cnt = 0;
for i = 1:size(rst,1)
    [temp index] = sort(rst(i,:),'descend');
    elems = index(1:top);
    if ~isempty(find(elems==i))
        cnt = cnt+1;
    end
end
avgTop=cnt/size(rst,1);
end