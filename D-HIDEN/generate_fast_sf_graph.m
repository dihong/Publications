% This function generates the scale-free graph can be computed by HIDEN
% efficiently.
function [Net,x,cost] = generate_fast_sf_graph(N,M,num_incoming_edges_per_node)
rng('default');
rng('shuffle');
    while 1
        Net = ba_sf_random_network(N,num_incoming_edges_per_node);%double(SFNG(N, 1, seed)); %scale free network.
        [x,cost,EXITFLAG] = HIDEN(Net,M);
        if numel(x)>0 % successful.
            break;
        end
    end
end

function [x,cost,EXITFLAG] = HIDEN(G,M)
ne = sum(G(:)); %#edges.
n = size(G,1); %#vertices.
A = zeros(ne,n);
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
options.MaxTime = 5;
[x,cost,EXITFLAG] = intlinprog([zeros(1,n) ones(1,ne)],1:n+ne,A,b,[],[],lb,ub,options);
if numel(x)>=n
    x = round(x(1:n)');
else
    x = [];
end
cost = round(cost);
end