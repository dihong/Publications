function model = prmi_random_forest_train(data,label,cfTrain,params,cfTest,nbags,sa,sb,probe,gallery)
% PRMI_RANDOM_FOREST_TRAIN Random Forest Ensemble Learning.
% model = prmi_random_forest_train(data,label,cfTrain,params,cfTest,nbags,sa,sb)
% performs random forest ensemble learning:
%
%   'model'   The learned ensemble model for classification.
%   'data'    The DxN training data matrix, one observation per column.
%   'label'   The 1xN training label corresponding to 'data'.
%   'cfTrain' The function handle of base classifier for training (i.e. @prmi_usa_train).
%   'params'  The parameters used to call the [classifier] function.
%   'cfTest'  The function handle of base classifier for testing (i.e. @prmi_usa_test).
%   'nbags'   The number of bags for Bagging ensemble.
%   'sa'      A (0,1) fraction number indicating the portion of attributes to be
%             sampled for each bag. For example, if sa = 0.6, then round (0.6*d)
%             attributes will be sampled randomly, where d is the dimension of samples. 
%   'sb'      [Optional] A (0,1) fraction number indicating the portition of samples
%             to be sampled for each bag. 
%   'probe'   [optional] A DxP matrix containing probe samples, one sample per column.
%   'gallery' [optional] A DxQ matrix containing gallery samples, one
%             sample per column. If both probe & gallery are present, then
%             it prints the rank-1 accuracy during training process.
% See also PRMI_RANDOM_FOREST_TEST.
%
% Contact: www.dihong.me

validateattributes(data,{'double'},{'nonsparse','finite','nonnan'},mfilename,'data',1);
if nargin==10
    validateattributes(probe,{'double'},{'nonsparse','finite','nonnan'},mfilename,'probe',9);
    validateattributes(gallery,{'double'},{'nonsparse','finite','nonnan'},mfilename,'gallery',10);
    figure;hold on;axis([1 nbags 0.0 1.0]); grid on;
    fprintf('\n');
end
validateattributes(params,{'cell'},{},mfilename,'params',4);
[d,n] = size(data);
k = round(sa*d);
if k<1 || k>d
    error('Invalid parameter [sa].');
elseif n ~= length(label)
    error('Dimensions of data and label are inconsistent.');
end
ul = unique(label);
ns = numel(ul);
for i = 1:nbags
    s1 = randperm(d);
    s1 = s1(1:k); %sampled attributes.
    if nargin==7
        s2 = round((ns-1)*rand(1,ns))+1; %sampled training samples.
    else
        s2 = randperm(ns);
        s2 = s2(1:round(sb*ns)); %sampled training samples.
    end
    L = [];
    D = [];
    tmp = data(s1,:);
    for j = 1:numel(s2)
        D = [D tmp(:,label==ul(s2(j)))];
        L = [L repmat(ul(s2(j)),1,sum(label==ul(s2(j))))];
    end
    models{i} = feval(cfTrain,D,L,params{:});
    attributes{i} = s1;
    
    model.models = models;
    model.attributes = attributes;
    model.cfTest = cfTest;
    if nargin==10 %print rank-1 accuracy.
        clear tmp;
        tmp.models = models(i);
        tmp.attributes = attributes(i);
        tmp.cfTest = cfTest;
        top1(i) = prmi_compute_top(prmi_random_forest_test(tmp,probe,gallery));
        top2(i) = prmi_compute_top(prmi_random_forest_test(model,probe,gallery));
        fprintf('Bag %d accuracy: %.4f. Total accuracy: %.4f\n',i,top1(i),top2(i));
        plot(1:i,top1,'b.-');hold on;plot(1:i,top2,'r.-');legend('Single Bag','All Bags');
        drawnow
    end
end
end