function joint_perb_dist_combination
fls = dir('joint_perb_dist/*.mat');
%[operation L/R S O]: operation = 1 for remove, 2 for add; L/R = 1 for neighbor of source vertice and L/R = 2 for sink vertice, L/R=3 for source vertice, 4 for sink vertice; S = 1->6 for status of neighbor. S = 1 to 3 means up,horizontal,and down of incoming edges, and 4 to 6 means outgoing edges. O = -1 means moving down, 0 means unchnaged, 1 means moving up.
src{1} = zeros(1,3);src{2} = zeros(1,3);
sink{1} = zeros(1,3);sink{2} = zeros(1,3);
nsrc{1} = zeros(6,3);nsrc{2} = zeros(6,3);
nsink{1} = zeros(6,3);nsink{2} = zeros(6,3);
% vertex (where "O" is the target vertex in the graph, "-" is incoming edge of target vertex, "=" is outoging edge).
%     6       3
%       \   //
%     5 - O = 2
%       /   \\
%     4       1
map_pos = [6 5 4 3 2 1];
for i = 1:length(fls)
    try
        load(['joint_perb_dist/' fls(i).name]);
    catch e
        delete(['joint_perb_dist/' fls(i).name]);
    end
    for j = 1:size(P,1)
        t = P(j,1:4);
        if t(2) == 1 %nsrc
            nsrc{t(1)}(map_pos(t(3)),t(4)+2) = nsrc{t(1)}(map_pos(t(3)),t(4)+2) + 1;
        elseif t(2)==2 %nsink
            nsink{t(1)}(map_pos(t(3)),t(4)+2) = nsink{t(1)}(map_pos(t(3)),t(4)+2) + 1;
        elseif t(2)==3 %src
            src{t(1)}(1,t(4)+2) = src{t(1)}(1,t(4)+2) + 1;
        else %sink
            sink{t(1)}(1,t(4)+2) = sink{t(1)}(1,t(4)+2) + 1;
        end
    end
end

for op = 1:2
    if sum(src{op}) ==0
        src{op} = [0 1 0];
    else
        src{op} = src{op}/sum(src{op});
    end
    if sum(sink{op}) ==0
        sink{op} = [0 1 0];
    else
        sink{op} = sink{op}/sum(sink{op});
    end
    for pos = 1:6
        if sum(nsrc{op}(pos,:)) ==0
            nsrc{op}(pos,:) = [0 1 0];
        else
            nsrc{op}(pos,:) = nsrc{op}(pos,:)/sum(nsrc{op}(pos,:));
        end
        if sum(nsink{op}(pos,:)) ==0
            nsink{op}(pos,:) = [0 1 0];
        else
            nsink{op}(pos,:) = nsink{op}(pos,:)/sum(nsink{op}(pos,:));
        end
    end
end
save('joint_perturb_dist','src','sink','nsrc','nsink');
end