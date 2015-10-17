function [feat age id]= extract_bif_feat(path,ext)
% path = 'E:\Computer Vision\Database\AgingData_FGNET\Images_norm_w260_crop_medium';%
% ext = 'BMP';
fls = dir([path '/' '*.' ext]);

if length(fls) == 0
    error('No image was found.');
end

nTheta = 12;
% 
% mex Pool.cpp;

FS1 = 4:2:14;
FS2 = 6:2:16;

Theta = pi*(0:nTheta-1)/nTheta;
Sigma1 = FS1/2;
Lamda1 = Sigma1 * 1.25;
Sigma2 = FS2/2;
Lamda2 = Sigma2 * 1.25;
% PoolWinSize = 8:4:36;
PoolWinSize = 6:3:21;

nBand = length(FS1);
% F = cell(nBand,nTheta);
feat = cell(1,length(fls));
age = feat;
id = feat;
tic
parfor indImg = 1:length(fls)
    nm = fls(indImg).name;
    I = imread([path '/' nm]);
    for i = 1:nBand
        for j = 1:nTheta
            I1 = gaborfilter(I,Sigma1(i),Lamda1(i),FS1(i),Theta(j));
            I2 = gaborfilter(I,Sigma2(i),Lamda2(i),FS2(i),Theta(j));
            Imax = max(I1,I2);
            feat{indImg} = [feat{indImg} ;Pool(Imax,PoolWinSize(i),size(Imax))]; % pool the features from the S1 layer.
        end
    end 
    id{indImg} = str2double(nm(1:3)); % the ID of the object
    age{indImg} = str2double(nm(5:6)); % the AGE of the object
end
toc
feat = cat(2,feat{:});
id = cat(2,id{:});
age = cat(2,age{:});
end




function [gabout G] = gaborfilter(I,sigma,lamda,win_size,theta,G)

% I = imread('E:\Computer Vision\Database\AgingData_FGNET\Images_norm_w260_crop\001A02.bmp');
% sigma = 2.8;
% lamda = 3.5;
% theta = 3*pi/4;
% win_size = 7;

if nargin > 5
    I = double(I);
    gabout = imfilter(I,G); 
    return;
end
gama = 0.3;
win_half = floor(win_size/2);
I = double(I);
[y x]= meshgrid(-win_half:win_half,-win_half:win_half);
xPrime = x*cos(theta) + y*sin(theta);
yPrime = y*cos(theta) - x*sin(theta);
G = exp(-.5*((xPrime/sigma).^2+(gama*yPrime/sigma).^2)).*cos(2*pi*xPrime/lamda);

gabout = imfilter(I,G); 

% gabout = (uint8(255*(gabout-min(gabout(:)))/(max(gabout(:))-min(gabout(:)))));
% 
% imshow(gabout);

end