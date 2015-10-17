%% This function extracts features based on encoded face images, then computes the matching scores 'rst' of testing faces based on LFDA subspace analysis.

function rst = test(winsize,radii,szpatch,sift)
nslice = 12;
d1 = 400;
d2 = 350;
d3 = d2-1;
sum_rst = 0;
for i = 1:length(winsize)
    for j = 1:length(radii)
        path1 = [pwd '\linear_encoded\' num2str(winsize{i}(1)) '_' num2str(winsize{i}(2)) '_' num2str(radii{j}) '\optical\'];
        path2 = [pwd '\linear_encoded\' num2str(winsize{i}(1)) '_' num2str(winsize{i}(2)) '_' num2str(radii{j}) '\infrared\'];
        fls1 = dir([path1 '\' '*.' 'bmp']);
        fls2 = dir([path2 '\' '*.' 'bmp']);
        sz_img = size(imread([path1 '\' fls1(1).name ]));
        parfor n = 1:length(fls1)
            feat1{n} = compute_hist(double(imread([path1 '\'  num2str(n) '.bmp' ])),szpatch,sift,sz_img);
            feat2{n} = compute_hist(double(imread([path2 '\'  num2str(n) '.bmp' ])),szpatch,sift,sz_img);
        end
        feat1 = cat(2,feat1{:});
        feat2 = cat(2,feat2{:});
        %% subspace analysis: LFDA
        [trainpairs testgallery testprobe] = gen_data(feat1, feat2, 1:1400, 1401:2800);
        clear feat1 feat2
        [trainpairs testgallery testprobe] = slice_data(trainpairs,testgallery,testprobe,nslice);
        dm = 0;
        parfor k = 1:nslice
            dm = dm + minMaxNorm(one_slice_lfda(trainpairs{k},testgallery{k},testprobe{k},d1,d2,d3));
        end
        %             ver = generateROC(dm,1e-3) % verification rate at far = 0.1%
        sum_rst = sum_rst + dm;
        fprintf('The accuracy for winsize = [%d %d], radii = %d: %.2f.\n',winsize{i}(1),winsize{i}(2),radii{j},computeTOP(dm));
        fprintf('The accumulated accuracy: %.2f.\n',computeTOP(sum_rst));
    end
end
fprintf('The final recognition accuracy: %.2f.\n',computeTOP(sum_rst));
rst = sum_rst;
save('rst','rst');
end

function [trainpairs testgallery testprobe] = gen_data(feat1, feat2, itrain, itest)
% feat1 is for the gallery[optical], feat is for the probe[infrared].
testgallery = feat1(:,itest);
testprobe = feat2(:,itest);
trainpairs = zeros(size(feat1,1),2*length(itrain));
trainpairs(:,1:2:end) = feat1(:,itrain);
trainpairs(:,2:2:end) = feat2(:,itrain);
end

function [sliced_traint1pairs sliced_testgallery sliced_testprobe] = slice_data(trainpairs,testgallery,testprobe,nslice)
d = round(length(trainpairs)/nslice);
for i = 1:nslice-1
    sliced_traint1pairs{i} = trainpairs((i-1)*d+1:i*d,:);
    sliced_testgallery{i} = testgallery((i-1)*d+1:i*d,:);
    sliced_testprobe{i} = testprobe((i-1)*d+1:i*d,:);
end
sliced_traint1pairs{nslice} = trainpairs((nslice-1)*d+1:end,:);
sliced_testgallery{nslice} = testgallery((nslice-1)*d+1:end,:);
sliced_testprobe{nslice} = testprobe((nslice-1)*d+1:end,:);
end

% This function return the recognition results [distance matrix] for one
% slice of the lfda.
function dm =one_slice_lfda(train,gallery,probe,d1,d2,d3)%LMCA: parrel.
%L2Norm
train = L2Norm(train);
gallery = L2Norm(gallery);
probe = L2Norm(probe);
%Train
[ProjectionMat m]=USA(train,d1,d2,d3);
%Match
gallery=ProjectionMat'*meanNorm(gallery,m);
probe=ProjectionMat'*meanNorm(probe,m);
dm = DistanceMatrix(gallery,probe,'cosine');
end