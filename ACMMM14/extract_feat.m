function feat = extract_feat(faces,type)

feat = cell(1,length(faces));

if strcmp(type,'dsift')
    descriptor = @dsift;
elseif strcmp(type,'mlbp')
    sift_fac = 0.5;
    szPatch = 14;
    scales = [1 3 5 7];
    feat = lbpExtract(szPatch,sift_fac,scales,faces);
    return;
elseif strcmp(type,'hog')
    step = 8;
    blk_size = 40;
    parfor indImg =1 : length(faces)
        if numel(faces{indImg})>0
            feat{indImg} = HOG(double(faces{indImg}), blk_size, step);
        end
    end
    feat = cat(2,feat{:});
    return;
end

parfor indImg = 1:length(faces)
    if numel(faces{indImg})>0
        feat{indImg} = descriptor(im2single(faces{indImg}));
    end
end
feat = cat(2,feat{:});
end



function f = dsift(I)
[pnt f] = vl_dsift(I,'step',8,'size',12,'Fast','FloatDescriptors');
f = double(f(:));
end