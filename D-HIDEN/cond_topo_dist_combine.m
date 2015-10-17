function cond_topo_dist_combine
fls = dir('cond_topo_dist/*.mat');
prob = ones(3,3,6);
for i = 1:length(fls)
    try
        load(['cond_topo_dist/' fls(i).name]);
    catch e
%         delete(['cond_topo_dist/' fls(i).name]);
        continue;
    end
    prob = prob + P;
end

for y = [1 3]
    for u = 1:6
        prob(:,y,u) = prob(:,y,u)/sum(prob(:,y,u));
    end
end
for u = 1:6
    prob(1,2,u) = 0;
    prob(2,2,u) = 1;
    prob(3,2,u) = 0;
end
dist = prob;
save('cond_topo_dist','dist');
end