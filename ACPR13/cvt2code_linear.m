
%% This function learns encoder and then converts face images into encoded images.

function cvt2code_linear(path1,path2,winsize,radii)
%% extract the pixel features.
fls = dir([path1 '\' '*.' 'bmp']);
sz_img = size(imread([path1 '\' fls(1).name ]));
parfor i = 1:200 % use first 200 faces as training.
    pf1{i} = extract_pixel_feat(double(imread([path1 '\' fls(i).name])),winsize,radii,sz_img);
    pf2{i} = extract_pixel_feat(double(imread([path2 '\' fls(i).name])),winsize,radii,sz_img);
end
%% train the feature encoder.
[w1 w2 b1 b2] = train_linear_cca(pf1,pf2,radii);
%% turn the images into encoded face images.
nx = floor(sz_img(1)/winsize(1));
ny = floor(sz_img(2)/winsize(2));
outpath1 = [pwd '\linear_encoded\' num2str(winsize(1)) '_' num2str(winsize(2)) '_' num2str(radii) '\optical\'];
if(~exist(outpath1,'dir'))
    mkdir(outpath1);
end
outpath2 = [pwd '\linear_encoded\' num2str(winsize(1)) '_' num2str(winsize(2)) '_' num2str(radii) '\infrared\'];
if(~exist(outpath2,'dir'))
    mkdir(outpath2);
end
parfor n = 1:length(fls)
    Icode1 = encode_one_image(extract_pixel_feat(double(imread([path1 '\'  num2str(n) '.bmp' ])),winsize,radii,sz_img),radii,sz_img,nx,ny,w1,b1,winsize);
    Icode2 = encode_one_image(extract_pixel_feat(double(imread([path2 '\'  num2str(n) '.bmp' ])),winsize,radii,sz_img),radii,sz_img,nx,ny,w2,b2,winsize);
    imwrite(Icode1,[outpath1 num2str(n) '.bmp']);
    imwrite(Icode2,[outpath2 num2str(n) '.bmp']);
end

end

function Icode = encode_one_image(pf,radii,sz_img,nx,ny,w,b,winsize)
    pf = slice_data(pf,radii);
    Icode = uint8(zeros(sz_img));
    Ibin = Icode;    
    for k = 1:size(pf,2) %each orientation
        for i = 1:nx %patch x
            for j = 1:ny %patch y
                ind = (j-1)*nx+i;
                code = encode_one_block(pf{ind,k},w{ind,k},b{ind,k},winsize);
                Ibin((i-1)*winsize(1)+1:i*winsize(1),(j-1)*winsize(2)+1:j*winsize(2)) = 255*code;
            end
        end
        Icode = 2*Icode + uint8(Ibin>0);
    end
end


function y = slice_data(x,r)
npatch = size(x,1)/8/(r+1);
y = cell(npatch,8);
for i = 1:npatch
    for j = 1:8
        y{i,j} = [y{i,j} x((i-1)*8*(r+1)+(j-1)*(r+1)+1:(i-1)*8*(r+1)+j*(r+1),:)];
    end
end
end

function code = encode_one_block(pf,w,b,winsize)
code = uint8(w'*pf-b>0);
code = reshape(code,[winsize(2) winsize(1)])';
end











