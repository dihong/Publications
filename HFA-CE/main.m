function main
%% Setup parallel computing
CoreNum=2;
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

%% Parameters
path_to_database = 'database/example';

%% Load data
trainlabels = [];
trainimages = {};
trainmodalities = [];
probelabels = [];
probeimages = {};
gallerylabels = [];
galleryimages = {};
lines = textread([path_to_database '/protocol.txt'], '%s', 'delimiter', '\n');
for lineno = 1:numel(lines)
    components = strsplit(lines{lineno}, '\t');
    type = components{1};
    modality = str2double(components{2});
    identity = str2double(components{3});
    im = imread([path_to_database '/' components{4}]);
    if strcmp(type, 'train')
        trainlabels = [trainlabels identity];
        trainimages = [trainimages im];
        trainmodalities = [trainmodalities modality];
    elseif strcmp(type, 'test')
        if modality == 1
            gallerylabels = [gallerylabels identity];
            galleryimages = [galleryimages im];
        elseif modality == 2
            probelabels = [probelabels identity];
            probeimages = [probeimages im];
        else
            error(['Undefined modality ' modality]);
        end
    else
        error(['Undefined data type ' type]);
    end
end

%% Train encoders
models = train_encoder(trainimages, trainlabels, trainmodalities);

%% Extract features
parfor i = 1:numel(trainimages)
    trainfeatures{i} = extract_feature(models,trainimages{i},trainmodalities(i));
end
trainfeatures = double(cat(2,trainfeatures{:}));
parfor i = 1:numel(probeimages)
    probefeatures{i} = extract_feature(models,probeimages{i},1);
end
probefeatures = double(cat(2,probefeatures{:}));
parfor i = 1:numel(galleryimages)
    galleryfeatures{i} = extract_feature(models,galleryimages{i},2);
end
galleryfeatures = double(cat(2,galleryfeatures{:}));

%% Train classification models
nslices = 1;
d1 = 5;
d2 = 5;
trainfeatures = prmi_slice(trainfeatures,nslices);
parfor i = 1:numel(trainfeatures)
    lfda{i} = prmi_usa_train(trainfeatures{i},trainlabels,d1,d2);
end

%% Subspace embedding
probefeatures = prmi_slice(probefeatures,nslices);
galleryfeatures = prmi_slice(galleryfeatures,nslices);
for i = 1:numel(probefeatures)
    probefeatures{i} = lfda{i}.P'*bsxfun(@minus,probefeatures{i},lfda{i}.M);
end
for i = 1:numel(galleryfeatures)
    galleryfeatures{i} = lfda{i}.P'*bsxfun(@minus,galleryfeatures{i},lfda{i}.M);
end
probefeatures = cat(1,probefeatures{:});
galleryfeatures = cat(1,galleryfeatures{:});

%% Matching faces to the gallery persons
index_of_best_matched = cosine_knn(probefeatures,galleryfeatures)
end


function ind = cosine_knn(probefeatures,galleryfeatures)
probefeatures = normc(probefeatures);
galleryfeatures = normc(galleryfeatures);
score = probefeatures'*galleryfeatures;
[val,ind] = max(score,[],2);
end


function ind = euclidean_knn(probefeatures,galleryfeatures)
score = prmi_compute_pairwise_distance(probefeatures,galleryfeatures);
[val,ind] = min(score,[],2);
end


function f = extract_feature(models,im,modality)
szpatch = 16;
sift = 0.5;
f0 = [];
if modality == 1
    for k = 1:numel(models)  % for each radius.
        Icode = encode_one_image(extract_pixel_feat(double(im),models{k}.radius,size(im),models{k}.Rx,models{k}.Ry),models{k}.radius,size(im),models{k}.w1,models{k}.B1,models{k}.kernel_type,models{k}.p11,models{k}.p21,models{k}.u1,models{k}.v1);
        sf = extract_feat_from_encoded(double(Icode),szpatch,sift,size(im),256);
        f0 = [f0;sf];
    end
elseif modality == 2
    for k = 1:numel(models)  % for each radius.
        Icode = encode_one_image(extract_pixel_feat(double(im),models{k}.radius,size(im),models{k}.Rx,models{k}.Ry),models{k}.radius,size(im),models{k}.w2,models{k}.B2,models{k}.kernel_type,models{k}.p12,models{k}.p22,models{k}.u2,models{k}.v2);
        sf = extract_feat_from_encoded(double(Icode),szpatch,sift,size(im),256);
        f0 = [f0;sf];
    end
else
    error ('undefined modality.')
end
f = f0;
end

function Icode = encode_one_image(pf,radii,sz_img,w,B,type,p1,p2,u,v)
Icode = 0;
M = size(B{1},2);
pf = [pf;pf(1,:)];
for i = 1:8
    Kbp = kernel(B{i},L2Norm(pf((i-1)*radii+1:i*radii+1,:)),type,p1{i},p2{i});
    y = w{i}'*Kbp-u{i}-sum(w{i})*sum(Kbp,1)/M+v{i};
    Icode = 2*Icode + (y>0);
end
Icode = uint8(reshape(Icode,sz_img));
end
