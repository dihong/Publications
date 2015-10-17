%% This function estimates the dynamic set D, as described in the Paper.
% linear dynamic system: x[t+1] = (1-alpha)*A*x[t]+alpha*b
% G[in]: the adjacent matrix of the graph before modification.
% Gp[in]: the adjacent matrix of the graph after modification.
% T[in]: the hierarchy assignment for G.
% ranking[out]: the Nx1 vector, vertex with smaller ranking is easier to be selected
% as member of dynamic set D.

% By Dihong, Jun 03, 2013.

function ranking = estimate_dynamic_set(G,Gp,T,alpha)
% [G,T,cost] = generate_fast_sf_graph(20,10);
% Z = random_graph_modification(G,'insert',1);
% Z = Z + random_graph_modification(G,'remove',1);
%
%compute the initial status based on perturbation.
N = size(G,1);
load('joint_perturb_dist'); %(2,4,7,3); %op=1->2,L/R=1->4,S=0->6,O=-1->1.
%[operation L/R S O]: operation = 1 for remove, 2 for add; L/R = 1 for neighbor of source vertice and L/R = 2 for sink vertice, L/R=3 for source vertice, 4 for sink vertice; S = 1->6 for status of neighbor. S = 1 to 3 means up,horizontal,and down of incoming edges, and 4 to 6 means outgoing edges. O = -1 means moving down, 0 means unchnaged, 1 means moving up.
num_status = 3*N;
b = zeros(num_status,1);
priors = cell(1,N);
for i = 1:N
    for j = 1:N
        if Gp(i,j)==0 && G(i,j)==1 %remove edge (i,j)
            op = 1;
        elseif Gp(i,j)==1 && G(i,j)==0 %insert edge (i,j)
            op = 2;
        else
            continue;
        end
        %source vertex
        p = src{op}';
        priors{i} = [priors{i} p];
        %sink vertex
        p = sink{op}';
        priors{j} = [priors{j} p];
        %neighbor of source
        for k = 1:N
            if G(i,k)==1
                if T(k)<T(i)
                    priors{k} = [priors{k} nsrc{op}(1,:)'];
                elseif T(k)==T(i)
                    priors{k} = [priors{k} nsrc{op}(2,:)'];
                else
                    priors{k} = [priors{k} nsrc{op}(3,:)'];
                end
            elseif G(k,i)==1
                if T(k)<T(i)
                    priors{k} = [priors{k} nsrc{op}(4,:)'];
                elseif T(k)==T(i)
                    priors{k} = [priors{k} nsrc{op}(5,:)'];
                else
                    priors{k} = [priors{k} nsrc{op}(6,:)'];
                end
            end
        end
        %neighbor of sink
        for k = 1:N
            if G(j,k)==1
                if T(k)<T(j)
                    priors{k} = [priors{k} nsink{op}(1,:)'];
                elseif T(k)==T(j)
                    priors{k} = [priors{k} nsink{op}(2,:)'];
                else
                    priors{k} = [priors{k} nsink{op}(3,:)'];
                end
            elseif G(k,j)==1
                if T(k)<T(j)
                    priors{k} = [priors{k} nsink{op}(4,:)'];
                elseif T(k)==T(j)
                    priors{k} = [priors{k} nsink{op}(5,:)'];
                else
                    priors{k} = [priors{k} nsink{op}(6,:)'];
                end
            end
        end
    end
end
for i = 1:N
    if size(priors{i},2)>0
        b((i-1)*3+1:i*3) = mean(priors{i},2);
    else
        b((i-1)*3+1:i*3) = [0 1 0];
    end
end
%compute linear transformation matrix A
A = zeros(num_status);
load('cond_topo_dist','dist');
Pu_x_y = dist; % Pu(i,j): probability that the target vertex has action i(down,same,up) when its neighbor who is connecting with status u has action j. u = {1,2,3,4,5,6} where 1->"out above", 2->"out same", 3->"out below", 4->"in above",...


for i = 1:N
    num_Ni = sum((Gp(i,:)'+Gp(:,i)>0.5)); %number of neighbors for vertex i.
    for ip = 1:N
        if Gp(i,ip)==1 %outgoing.
            base = 0;
        elseif Gp(ip,i)==1 %incoming.
            base = 3;
        else
            continue;
        end
        if T(i)>T(ip)
            stat = base + 1;
        elseif T(i)==T(ip)
            stat = base + 2;
        else
            stat = base + 3;
        end
        for k = 1:3
            for kp = 1:3
                A((i-1)*3+k,(ip-1)*3+kp) = Pu_x_y(k,kp,stat)/num_Ni;
            end
        end
    end
end

if alpha==0
    [V,D] = eig(A);
    [~,ind] = sort(abs(diag(D)),'descend');
    x = V(:,ind(1));
else
    x = (eye(num_status)-(1-alpha)*A)\b*alpha;
end

for j = 1:numel(x)/3
    x((j-1)*3+1:j*3) = x((j-1)*3+1:j*3)/sum(x((j-1)*3+1:j*3));
end

% x = b;
x = reshape(x,3,[]);
if nargin==5
    x(2,:) = [];
    xs = 1-max(x,[],1);
else
    xs = x(2,:); %probability of staying the same level.
end
[~,ranking] = sort(xs,'ascend');
end