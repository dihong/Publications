function [w1 w2 b1 b2] = train_linear_cca(pixel_feat_1,pixel_feat_2,radii)
pixel_feat_1 = slice_data(pixel_feat_1,radii);
pixel_feat_2 = slice_data(pixel_feat_2,radii);
for i = 1:size(pixel_feat_1,1) %npatch.
    for j = 1:8 %orientation.
        [w1{i,j} w2{i,j} b1{i,j} b2{i,j}] = linear_cca(pixel_feat_1{i,j},pixel_feat_2{i,j});
    end
end
end

function y = slice_data(x,r)
npatch = size(x{1},1)/8/(r+1);
y = cell(npatch,8);
for n = 1:length(x)
    for i = 1:npatch
        for j = 1:8
            y{i,j} = [y{i,j} x{n}((i-1)*8*(r+1)+(j-1)*(r+1)+1:(i-1)*8*(r+1)+j*(r+1),:)];
        end
    end
end
end


function [w1 w2 b1 b2] = linear_cca(pixel_feat_1,pixel_feat_2)
[A B r U V] = canoncorr(pixel_feat_1',pixel_feat_2');
w1 = A(:,1);
w2 = B(:,1);
b1 = mean(U(:,1));
b2 = mean(V(:,1));
end