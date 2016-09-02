function W = prmi_ltsa(X,pid,age,d)
% PRMI_LTSA Linear Topological Structure Analysis
% W = prmi_ltsa(X,pid,age,d) performs linear topological strucutre analysis
% for data matrix X
% 'X' PxN data matrix with each column containing one observation.
% 'pid' 1xN integer vector where pid(n) is the ID of the n-th observation.
% 'age' 1xN integer vector where age(n) is the age of the n-th observation.
% 'd' an indeger of the dimension of the subspace.
% 'W' is a Pxd subspace projection matrix.
%
% Contact: www.dihong.me
G = unique(pid);
D = size(X);
DX = [];
CX = [];
for i = 1:numel(G)
    Xg = X(:,pid==G(i)); %all the samples belonging to the same pid.
    Cg = age(pid==G(i)); %the corresponding age.
    C = unique(Cg);
    for j = 1:numel(C)
        Xgc{j} = Xg(:,Cg==C(j));
    end
    num_dx = 0;
    for j = 1:numel(C)
        for k = j+1:numel(C)
            num_dx = num_dx + size(Xgc{j},2)*size(Xgc{k},2);
        end
    end
    Dx = zeros(D,num_dx);
    Cx = zeros(1,num_dx);
    cnt = 1;
    for j = 1:numel(C)
        for k = j+1:numel(C)
            nt = size(Xgc{k},2);
            for s = 1:size(Xgc{j},2)
                Dx(:,cnt:cnt+nt-1) = bsxfun(@minus,Xgc{k},Xgc{j}(:,s));
                Cx(cnt:cnt+nt-1) = (C(j)-C(k));
            end
            cnt = cnt + nt;
        end
    end
    DX = [DX Dx];
    CX = [CX Cx];
end
CX = abs(CX);
DX = bsxfun(@times,DX,CX);
A = DX*DX';
[V,D] = eig(A);
[~,ind] = sort(diag(D),'descend');
W = V(:,ind(1:d));
end