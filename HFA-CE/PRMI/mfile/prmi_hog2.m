function F = prmi_hog2(I)
scale = [1 .75 .5];
szpatch = 48;
sift = 0.15;
F = [];
for s = scale
    szImage=size(imresize(I,s));
    f = HOG2(imresize(I,s),szpatch,sift,szImage);
    F = [F;f];
end
end