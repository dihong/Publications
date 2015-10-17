function joint_perb_dist
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
for i = 1:1e6
    [Net,x,cost] = generate_fast_sf_graph(N,M);
    run_one_graph(Net,M,x,cost);
end
end


function run_one_graph(G,M,T,c)
%Remove one 'non-conflicting edge' (downward):
MaxRemovedEdge = 10; %the maximum number of removed edge for this graph.
MaxAddedEdge = 10; %the maximum number of added edge for this graph.
N = size(G,1); %number of vertices.
NE = sum(G(:)); %number of edges.
cnt = 0;
cnt2 = 0;
output = []; % [operation L/R S O]: operation = 1 for remove, 2 for add; L/R = 1 for neighbor of source vertice and L/R = 2 for sink vertice, L/R=3 for source vertice, 4 for sink vertice; S = 1->6 for status of neighbor. S = 1 to 3 means up,horizontal,and down of incoming edges, and 4 to 6 means outgoing edges. O = -1 means moving down, 0 means unchnaged, 1 means moving up.
for i = 1:N
    for j = 1:N
        if G(i,j)==1 && T(i)>T(j) && cnt<MaxRemovedEdge %non-conflicting edge.
            Gt = G;
            Gt(i,j) = 0;
            [x,cost,EXITFLAG] = HIDEN(Gt,M); %run the HIDEN based on the perturbed graph.
            if EXITFLAG~=1
                continue;
            end
            for k = 1:N %for each incoming edge of i.
                if Gt(k,i)==1
                    op = 1; %remove
                    LR = 1; %neighbor of source
                    if T(k)>T(i)
                        S = 1;
                    elseif T(k)==T(i)
                        S = 2;
                    else
                        S = 3;
                    end
                    if cost==c
                        O = 0; %unchanged
                    else
                        if x(k)>T(k)
                            O = 1;
                        elseif x(k)<T(k)
                            O = -1;
                        else
                            O = 0;
                        end
                    end
                    output = [output; op LR S O N NE c cost];
                end
            end
            for k = 1:N %for each outgoing edge of i.
                if Gt(i,k)==1
                    op = 1; %remove
                    LR = 1; %neighbor of source
                    if T(k)>T(i)
                        S = 4;
                    elseif T(k)==T(i)
                        S = 5;
                    else
                        S = 6;
                    end
                    if cost==c
                        O = 0; %unchanged
                    else
                        if x(k)>T(k)
                            O = 1;
                        elseif x(k)<T(k)
                            O = -1;
                        else
                            O = 0;
                        end
                    end
                    output = [output; op LR S O N NE c cost];
                end
            end
            for k = 1:N %for each incoming edge of j.
                if Gt(k,j)==1
                    op = 1; %remove
                    LR = 2; %neighbor of sink
                    if T(k)>T(j)
                        S = 1;
                    elseif T(k)==T(j)
                        S = 2;
                    else
                        S = 3;
                    end
                    if cost==c
                        O = 0; %unchanged
                    else
                        if x(k)>T(k)
                            O = 1;
                        elseif x(k)<T(k)
                            O = -1;
                        else
                            O = 0;
                        end
                    end
                    output = [output; op LR S O N NE c cost];
                end
            end
            for k = 1:N %for each outgoing edge of j.
                if Gt(j,k)==1
                    op = 1; %remove
                    LR = 2; %neighbor of sink
                    if T(k)>T(j)
                        S = 4;
                    elseif T(k)==T(j)
                        S = 5;
                    else
                        S = 6;
                    end
                    if cost==c
                        O = 0; %unchanged
                    else
                        if x(k)>T(k)
                            O = 1;
                        elseif x(k)<T(k)
                            O = -1;
                        else
                            O = 0;
                        end
                    end
                    output = [output; op LR S O N NE c cost];
                end
            end
            %source
            op = 1;
            LR = 3;
            S = 0; %no status for source/sink.
            if cost==c
                O = 0; %unchanged
            else
                if x(i)>T(i)
                    O = 1;
                elseif x(i)<T(i)
                    O = -1;
                else
                    O = 0;
                end
            end
            output = [output; op LR S O N NE c cost];
            %sink
            op = 1;
            LR = 4;
            S = 0; %no status for source/sink.
            if cost==c
                O = 0; %unchanged
            else
                if x(j)>T(j)
                    O = 1;
                elseif x(j)<T(j)
                    O = -1;
                else
                    O = 0;
                end
            end
            output = [output; op LR S O N NE c cost];
            cnt = cnt + 1;
        end
        if G(i,j)==0 && T(i)<=T(j) && cnt2<MaxAddedEdge %non-existing conflicting edge.
            Gt = G;
            Gt(i,j) = 1; %add an edge
            [x,cost,EXITFLAG] = HIDEN(Gt,M); %run the HIDEN based on the perturbed graph.
            if EXITFLAG~=1
                continue;
            end
            op = 2; %add
            for k = 1:N %for each incoming edge of i.
                if Gt(k,i)==1
                    LR = 1; %neighbor of source
                    if T(k)>T(i)
                        S = 1;
                    elseif T(k)==T(i)
                        S = 2;
                    else
                        S = 3;
                    end
                    if cost>c
                        O = 0; %unchanged
                    else
                        if x(k)>T(k)
                            O = 1;
                        elseif x(k)<T(k)
                            O = -1;
                        else
                            O = 0;
                        end
                    end
                    output = [output; op LR S O N NE c cost];
                end
            end
            for k = 1:N %for each outgoing edge of i.
                if Gt(i,k)==1
                    LR = 1; %neighbor of source
                    if T(k)>T(i)
                        S = 4;
                    elseif T(k)==T(i)
                        S = 5;
                    else
                        S = 6;
                    end
                    if cost>c
                        O = 0; %unchanged
                    else
                        if x(k)>T(k)
                            O = 1;
                        elseif x(k)<T(k)
                            O = -1;
                        else
                            O = 0;
                        end
                    end
                    output = [output; op LR S O N NE c cost];
                end
            end
            for k = 1:N %for each incoming edge of j.
                if Gt(k,j)==1
                    LR = 2; %neighbor of sink
                    if T(k)>T(j)
                        S = 1;
                    elseif T(k)==T(j)
                        S = 2;
                    else
                        S = 3;
                    end
                    if cost>c
                        O = 0; %unchanged
                    else
                        if x(k)>T(k)
                            O = 1;
                        elseif x(k)<T(k)
                            O = -1;
                        else
                            O = 0;
                        end
                    end
                    output = [output; op LR S O N NE c cost];
                end
            end
            for k = 1:N %for each outgoing edge of j.
                if Gt(j,k)==1
                    LR = 2; %neighbor of sink
                    if T(k)>T(j)
                        S = 4;
                    elseif T(k)==T(j)
                        S = 5;
                    else
                        S = 6;
                    end
                    if cost>c
                        O = 0; %unchanged
                    else
                        if x(k)>T(k)
                            O = 1;
                        elseif x(k)<T(k)
                            O = -1;
                        else
                            O = 0;
                        end
                    end
                    output = [output; op LR S O N NE c cost];
                end
            end
            %source
            LR = 3;
            S = 0; %no status for source/sink.
            if cost>c
                O = 0; %unchanged
            else
                if x(i)>T(i)
                    O = 1;
                elseif x(i)<T(i)
                    O = -1;
                else
                    O = 0;
                end
            end
            output = [output; op LR S O N NE c cost];
            %sink
            LR = 4;
            S = 0; %no status for source/sink.
            if cost>c
                O = 0; %unchanged
            else
                if x(j)>T(j)
                    O = 1;
                elseif x(j)<T(j)
                    O = -1;
                else
                    O = 0;
                end
            end
            output = [output; op LR S O N NE c cost];
            cnt2 = cnt2 + 1;
        end
    end
end
P = output;
id = 1+numel(dir('joint_perb_dist/*.mat'));
save(['joint_perb_dist/' num2str(id)],'P');
end


function [x,cost,EXITFLAG] = HIDEN(G,M)
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
[x,cost,EXITFLAG] = intlinprog([zeros(1,n) ones(1,ne)],1:n+ne,A,b,[],[],lb,ub,options);
x = int32(x(1:n)');
cost = round(cost);
end