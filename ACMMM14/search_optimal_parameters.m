function search_optimal_parameters
path = '../Data/AgingData_FGNET/Images_norm_w260_crop_medium';
ext = 'bmp';
[feat age id]= extract_feat(path,ext,'dsift');  %'hog','dsift','bif'
[P M] = PCA(feat,0.99);
feat = P'*bsxfun(@minus,feat,M);
feat = L2Norm(feat);

param = cell(1,4);
step = cell(1,4);
param{1} = 0.8; step{1} = 0.1; %sigma
param{2} = 1; step{2} = 0.1; %theta1
param{3} = 1; step{3} = 0.1; %theta2
param{4} = 1; step{4} = 0.1; %phi2
cnt = 1;
tic
E0 = function_to_be_minimized(feat,age,id,param{:});
clf;
hold on;
param_hist = param;
cmap = hsv(length(param));  %# Creates a 6-by-3 set of colors from the HSV colormap
for i = 1:length(param)
    signs{i} = 1;
end
while 1
    %plot.
    for i = 1:length(param)
        plot(param_hist{i},'o-','Color',cmap(i,:));
    end
    legend('sigma','t1','t2','p2');
    grid on;
    pause(0.1);
    %
    for i = 1:length(param)
        param{i} = param{i} + signs{i}*step{i};
        E1 = function_to_be_minimized(feat,age,id,param{:});
        if E1>E0
            param{i} = param{i} - signs{i}*step{i};
            signs{i} = signs{i}*(-1);
            if cnt>2 && param{i}(end)==param{i}(end-1)
                step{i} = step{i} / 5;
            end
        else
            E0 = E1;
            step{i} = step{i}*2;
        end
        fprintf('[%d] sigma=%.2f, theta1 = %.2f, theta2 = %.2f, phi2 = %.2f, E = %.4f, time elapsed: %.2f\n',cnt,param{1},param{2},param{3},param{4},E0,toc);
        param_hist{i} = [param_hist{i} param{i}];
    end
    cnt = cnt + 1;
end
end