%% This function runs the HIDEN algorithm locally based on given dynamic set.
% Gp[in]: NxN adjacent matrix where N is the number of vertices (modified network).
% T[in]: Nx1 vector where T(n) is the hierarchy assignment of the n-th vertex.
% D[in]: Kx1 vector where D(k) is the vertices in the "dynamic set".
% M[in]: The maximum allowed number of hierarchies.
% assignment[out]: the computed hierarchy assignment for G.
% cost[out]: a scalar for the penalty cost of 'assignment'.

% By Dihong June 03, 2014.

function [assignment,cost] = local_hiden(Gp,T,D,M)
%Create subgraph
C = []; %the set C in the paper.
Tc = []; %the corresponding assignment for C.
N = size(Gp,1); % #vertices.
K = numel(D);
for i = 1:K
    for j = 1:N
        if (Gp(D(i),j)==1 || Gp(j,D(i))==1) && numel(find(D==j)) ==0 % neighbor j not in D
            if numel(find(C==j)) == 0
                C = [C; j];
                Tc = [Tc T(j)];
            end
        end
    end
end
Vs = [D(:);C];
Ns = numel(Vs); %#vertices in the subgraph
Gs = zeros(Ns); %the induced subgraph
for i = 1:Ns
    for j = 1:Ns
        if i>K && j>K,continue,end %these edges are C-C.
        Gs(i,j) = Gp(Vs(i),Vs(j));
    end
end
%HIDEN
ne = sum(Gs(:)); % #edges in the subgraph.
nd = numel(D); % #elements in the D.
A = zeros(ne,nd);
b = -1*ones(ne,1);
cnt = 1;
for i = 1:Ns
    for j = 1:Ns
        if Gs(i,j)==1 && i<=K && j<=K %D-D
            A(cnt,j) = 1;
            A(cnt,i) = -1;
            cnt = cnt + 1;
        elseif Gs(i,j)==1 && i>K && j<=K %C-D
            A(cnt,j) = 1;
            b(cnt) = b(cnt) + Tc(i-K);
            cnt = cnt + 1;
        elseif Gs(i,j)==1 && i<=K && j>K %D-C
            A(cnt,i) = -1;
            b(cnt) = b(cnt) - Tc(j-K);
            cnt = cnt + 1;
        end
    end
end
A = [A -M*eye(ne)];
lb = [ones(1,nd) zeros(1,ne)];
ub= [M*ones(1,nd) ones(1,ne)];
options = optimoptions('intlinprog');
options.Display = 'none';
options.MaxTime = 5;
x = intlinprog([zeros(1,nd) ones(1,ne)],1:nd+ne,A,b,[],[],lb,ub,options);
assignment = T;
if numel(x)>=nd
    x = round(x(1:nd)'); %assignment for the dynamic set D.
    assignment(D) = x;
end
cost = 0;
%counting the number of conflicting edges.
for i = 1:N
    for j = 1:N
        if Gp(i,j)==1 && assignment(i)<=assignment(j)
            cost = cost + 1;
        end
    end
end
end