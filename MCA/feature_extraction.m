function feat = feature_extraction(path,ext,type)
fls = dir([path '/' '*.' ext]);

if isempty(fls)
    error('No image was found.');
end

feat = cell(1,length(fls));

if strcmp(type,'hog')
    descriptor = @hog;
elseif strcmp(type,'dsift')
    descriptor = @dsift;
else
    error('undefined descriptor.');
end

parfor indImg = 1:length(fls)
    nm = fls(indImg).name;
    I = imread([path '/' nm]);
    feat{indImg} = descriptor(im2single(I));
end
feat = cat(2,feat{:});
end

function f = dsift(I)
[pnt f] = vl_dsift(I,'step',14,'size',14,'Fast','FloatDescriptors');
f = double(f(:));
end

function f = hog(I)
f = HOG(double(histeq(I)),64,20);
end