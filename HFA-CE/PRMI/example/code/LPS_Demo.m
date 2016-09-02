function LPS_Demo
%% Model training.
mex '../../feature_extraction/LPS_train.cpp'
clc
load('ferect','image1','image2');
ncode = 8;
radii = [1 3];
alpha = 0.5;
tic
model = LPS_train(image1,image2,ncode,radii,alpha) %images1, images2, ncode, radii, alpha
save('model','model');
toc
%% Model testing.
mex '../../feature_extraction/LPS_test.cpp'
clc
load('ferect','image1','image2');
load('model','model');
I = image1{1};
winsize = 16;
step = 8;
for i = 1:1000
    [f{i},landmarks{i},icode{i}] = LPS_test(I,winsize,step,[],model);
end
clear f;
I = uint8(255*(max(icode{1}(:))-icode{1})/(max(icode{1}(:))-min(icode{1}(:))));
hFig = figure(1); set(hFig, 'Position', [400 400 1200 400]);
subplot(1,3,1);imshow(image1{1}); title('Original Image');
subplot(1,3,2);imshow(I); title('Encoded Image');
subplot(1,3,3); 
[y,x] = hist(icode{1}(:),model(1).ncode);
bar(x,y,'r'); title('Code emergence of encoded image');

%% Call the lib.
load('ferect','image1','image2');
model = prmi_lps_train(image1(1:10),image2(1:10),'ncode',256,'scales',[1 3],'alpha',0.4);
[Ft,F,P,Icode] = prmi_lps(model,image1{1},'winsize',20,'step',20);
imshow(uint8(Icode{1}))
end