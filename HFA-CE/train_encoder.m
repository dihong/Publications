%% This function trains feature encoder.
function models = train_encoder(trainimages,trainlabels,trainmodalities)
%% Parameters
radii = [3,5,7,9];
kernel_type = 'Linear';
%% Construct training pairs.
subject_id = unique(trainlabels);
samples_1 = [];
samples_2 = [];
for k = 1:numel(subject_id)
    samples = find(trainlabels==subject_id(k));
    if numel(samples) > 1
        samples_1 = [samples_1,samples(1)];
        samples_2 = [samples_2,samples(end)];
        if trainmodalities(samples(1)) ~= 1
            error('The first sample must be in modality 1')
        end
        if trainmodalities(samples(end)) ~= 2
            error('The last sample must be in modality 2')
        end
    end
end
%%
sz_img = size(trainimages{1});
for k = 1:numel(radii)
    % extract pixel features.
    [Rx, Ry] = sampling_pattern(radii(k));
    parfor i = 1:numel(samples_1)
        pf1{i} = select_pixels(extract_pixel_feat(double(trainimages{samples_1(i)}),radii(k),sz_img,Rx,Ry));
        pf2{i} = select_pixels(extract_pixel_feat(double(trainimages{samples_2(i)}),radii(k),sz_img,Rx,Ry));
    end
    pf1 = cat(2,pf1{:});
    pf2 = cat(2,pf2{:});
    pf1 = group_feature(pf1,radii(k));
    pf2 = group_feature(pf2,radii(k));
    T = 100; %number of samples used as bases.
    for i = 1:8 %orientation.
        [w1{i}, w2{i}, B1{i}, B2{i}, p11{i}, p12{i}, p21{i}, p22{i}, Kbz1{i}, Kbz2{i}] = optimize(pf1{i},pf2{i},kernel_type,T);
        u1{i} = sum(Kbz1{i}*w1{i})/T;
        v1{i} = sum(Kbz1{i}(:))*sum(w1{i})/(T^2);
        u2{i} = sum(Kbz2{i}*w2{i})/T;
        v2{i} = sum(Kbz2{i}(:))*sum(w2{i})/(T^2);
    end
    models{k}.w1 = w1;
    models{k}.w2 = w2;
    models{k}.B1 = B1;
    models{k}.B2 = B2;
    models{k}.p11 = p11;
    models{k}.p12 = p12;
    models{k}.p21 = p21;
    models{k}.p22 = p22;
    models{k}.u1 = u1;
    models{k}.v1 = v1;
    models{k}.u2 = u2;
    models{k}.v2 = v2;
    models{k}.raius = radii(k);
    models{k}.Rx = Rx;
    models{k}.Ry = Ry;
    models{k}.kernel_type = kernel_type;
    models{k}.radius = radii(k);
    clear pf1 pf2;
end
end

function selected_pf = select_pixels(pf)
total = size(pf, 2);
sel = randperm(total);
sel = sel(1:10*10);
selected_pf = pf(:, sel);
end

function [Rx,Ry] = sampling_pattern(radii)
Rx = zeros(1,8*radii);
Ry = Rx;
for i = -1:8*radii-2
    if i<radii
        Rx(i+2) = radii;
        Ry(i+2) = i;
    elseif i<3*radii+1
        Rx(i+2) = 2*radii-i;
        Ry(i+2) = radii;
    elseif i<5*radii
        Rx(i+2) = -radii;
        Ry(i+2) = 4*radii-i;
    else
        Rx(i+2) = i-6*radii;
        Ry(i+2) = -radii;
    end
end
end


function [w1, w2, B1, B2, p11, p12, p21, p22, Kbz1, Kbz2] = optimize(pf1,pf2,kernel_type,T)
sel = randperm(size(pf1,2));
B1 = pf1(:,sel(1:T));
B2 = pf2(:,sel(1:T));
p11 = []; p12 = []; p21 = []; p22 = []; % parameters for kernel. p11, p21 for modality 1.
if strcmp(kernel_type,'Linear')
    Kbz1 = kernel(B1,B1,kernel_type);
    Kbz2 = kernel(B2,B2,kernel_type);
elseif strcmp(kernel_type,'Gaussian')
    g1 = 1/2;
    g2 = 1/2;
    Kbz1 = kernel(B1,B1,kernel_type,g1);
    Kbz2 = kernel(B2,B2,kernel_type,g2);  
    p11 = g1;
    p12 = g2;
elseif strcmp(kernel_type,'Sigmoid')
    p11 = 5;
    p21 = 0;
    p12 = 5;
    p22 = 0;
    Kbz1 = kernel(B1,B1,kernel_type,p11,p21);
    Kbz2 = kernel(B2,B2,kernel_type,p12,p22);
else
    fprintf('Undefined kernel type: %s',type);
    error('');
end
xi = cross_validata_xi(pf1,pf2,kernel_type,T,Kbz1,Kbz2,B1,B2,p11,p12,p21,p22);
Kzz1 = Kbz1;
Kzz2 = Kbz2;
K1 = centered_kernel(B1,pf1,Kbz1,Kzz1,kernel_type,p11,p21);
K2 = centered_kernel(B2,pf2,Kbz2,Kzz2,kernel_type,p12,p22);
K12 = K1*K2';
K11 = K1*K1';
K22 = K2*K2';
Z = zeros(T);
A = [Z K12;K12' Z];
B = [K11 Z;Z K22]+xi*eye(2*T);
[v, d] = eig(A,B,'chol');
[val, ind] = sort(diag(d),'descend');
v = v(:,ind(1));
w1 = v(1:T);
w2 = v(1+T:end);
s1 = sqrt(size(pf1,2)/(w1'*K11*w1));
s2 = sqrt(size(pf2,2)/(w2'*K22*w2));
w1 = s1*w1;
w2 = s2*w2;
end

function xi = cross_validata_xi(pf1,pf2,kernel_type,T,Kbz1,Kbz2,B1,B2,p11,p12,p21,p22)
% split the training data
N = size(pf1, 2);
Nt = round(N*0.6);
rind = randperm(N);
Xt = pf1(:,rind(1:Nt)); Xv = pf1(:,rind(1+Nt:end));
Yt = pf2(:,rind(1:Nt)); Yv = pf2(:,rind(1+Nt:end));
% evaluate K1, K2 for the training data
Kzz1 = Kbz1;
Kzz2 = Kbz2;
K1 = centered_kernel(B1,Xt,Kbz1,Kzz1,kernel_type,p11,p21);
K2 = centered_kernel(B2,Yt,Kbz2,Kzz2,kernel_type,p12,p22);
K12 = K1*K2';
K11 = K1*K1';
K22 = K2*K2';
Z = zeros(T);
A = [Z K12;K12' Z];
% evaluate K1, K2 for the validation data
K1v = centered_kernel(B1,Xv,Kbz1,Kzz1,kernel_type,p11,p21);
K2v = centered_kernel(B2,Yv,Kbz2,Kzz2,kernel_type,p12,p22);
% Loop through different xi:
xi = 1e3;
copt = [];
xopt = [];
while (xi>1e-3)
    % evaluate optimal w1, w2 for current xi.
    B = [K11 Z;Z K22]+xi*eye(2*T);
    [v, d] = eig(A,B,'chol');
    [val, ind] = sort(diag(d),'descend');
    v = v(:,ind(1));
    w1 = v(1:T);
    w2 = v(1+T:end);
    s1 = sqrt(size(pf1,2)/(w1'*K11*w1));
    s2 = sqrt(size(pf2,2)/(w2'*K22*w2));
    w1 = s1*w1;
    w2 = s2*w2;
    % evaluate the correlation of [w1 w2] w.r.t. K1v and K2v
    xopt = [xopt xi];
    copt = [copt w1'*K1v*K2v'*w2];
    xi = xi/10;
end
[val, ind] = max(copt);
xi = xopt(ind);
end

function K = centered_kernel(B,X,Kbz,Kzz,type,p1,p2)
K = kernel(B,X,type,p1,p2);
M = size(B,2);
K = bsxfun(@minus,bsxfun(@minus,K,sum(Kbz,2)/M),sum(K,1)/M)+sum(Kzz(:))/(M^2);
end

function y = group_feature(x,r)
y = cell(1,8);
x = [x;x(1,:)];
for i = 1:8
    y{i} = L2Norm(x((i-1)*r+1:i*r+1,:));
end
end

