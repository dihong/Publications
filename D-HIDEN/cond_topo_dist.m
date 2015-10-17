%% This function computes the conditional distribution of P(x,y,u).
% x: 1->"move down", 2->"stay same", 3->"move up" (target vertex).
% y: 1->"move down", 2->"stay same", 3->"move up" (neighbor vertex).
% u: {1,2,3,4,5,6} representing the location between target and neighbor
% vertex (where "O" is the target vertex in the graph, "-" is incoming edge of target vertex, "=" is outoging edge).
%     6       3
%       \   //
%     5 - O = 2
%       /   \\
%     4       1
% P(x,y,u) represents the probability that the target has movement x on the
% condition that neithbor has action y who connecting by u.
function cond_topo_dist
CoreNum=12;
MATLABPOOL=1;
if ~MATLABPOOL && (matlabpool('size')>0)
    matlabpool close;
end
if MATLABPOOL
    if(matlabpool('size')~=CoreNum && matlabpool('size')>0)
        matlabpool close;
        matlabpool('open',CoreNum) %open local.
    elseif(matlabpool('size')<1)
        matlabpool('open',CoreNum) %open local.
    end
end
%generate scale-free graph.
N = 50;
M = 10; %number of maximum layers.
tic
parfor i = 1:1e6
    [Net,x] = generate_fast_sf_graph(N,M);
    run_one_graph(Net,M,x);
end
toc
end


function [x,cost,EXITFLAG] = HIDEN(G,M,Aeq,beq)
ne = sum(G(:)); %#edges.
n = size(G,1); %#vertices.
A = zeros(n);
b = -1*ones(ne,1);
cnt = 1;
for i = 1:n
    for j = 1:n
        if G(i,j)==1
            A(cnt,j) = 1;
            A(cnt,i) = -1;
            cnt = cnt + 1;
        end
    end
end
A = [A -M*eye(ne)];
lb = [ones(1,n) zeros(1,ne)];
ub= [M*ones(1,n) ones(1,ne)];
options = optimoptions('intlinprog');
options.Display = 'none';
options.MaxTime = 2;
[x,cost,EXITFLAG] = intlinprog([zeros(1,n) ones(1,ne)],1:n+ne,A,b,Aeq,beq,lb,ub,options);
x = int32(x(1:n)');
cost = round(cost);
end


function P = run_one_graph(G,M,T)
P = zeros(3,3,6);
N = size(G,1); %number of vertices.
NE = sum(G(:)); %#edges.
num_vertices_changed_per_graph = 5;
rind = randperm(N);
cnt = 0;
i = 1;
while i <= N && cnt < num_vertices_changed_per_graph
    v = rind(i); %selected vertex.
    a1 = T(v); %original assignment.
    a2 = randperm(M); %select a2 ~= a1.
    if a2(1)==a1
        a2 = a2(2);
    else
        a2 = a2(1);
    end
    %run the hiden again by fixing a(v) = a2.
    Aeq = zeros(1,N+NE); Aeq(v) = 1;
    beq = a2;
    [T2,c,ef] = HIDEN(G,M,Aeq,beq);
    if ef~= 1
        continue;
    end
    cnt = cnt + 1;
    %neighbor action: either move up or down.
    if T(v)<T2(v), y = 3; else y = 1;end
    %target action:
    for j = 1:N
        if G(j,v)==1 %edge from target j to neithbor v.
            if T(v)>T(j) % target above neighbor
                u = 1;
            elseif T(v)==T(j)
                u = 2;
            else
                u = 3;
            end
            if T(j)<T2(j), x = 3; elseif T(j)==T2(j), x = 2; else x = 1; end
            P(x,y,u) = P(x,y,u) + 1;
        end
        if G(v,j)==1
            if T(v)>T(j) % target above neighbor
                u = 4;
            elseif T(v)==T(j)
                u = 5;
            else
                u = 6;
            end
            if T(j)<T2(j), x = 3; elseif T(j)==T2(j), x = 2; else x = 1; end
            P(x,y,u) = P(x,y,u) + 1;
        end
    end
    i = i + 1;
end
% for y = [1 3]
%     for u = 1:6
%         P(:,y,u) = P(:,y,u)/sum(P(:,y,u));
%     end
% end
% for u = 1:6
%     P(1,2,u) = 0;
%     P(2,2,u) = 1;
%     P(3,2,u) = 0;
% end
id = 1+numel(dir('cond_topo_dist/*.mat'));
save(['cond_topo_dist/' num2str(id)],'P');
end