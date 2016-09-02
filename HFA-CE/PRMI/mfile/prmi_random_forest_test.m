function scores = prmi_random_forest_test(model,probe,gallery)
% PRMI_RANDOM_FOREST_TEST Random Forest Ensemble Testing.
% scores = prmi_random_forest_test(model,probe,gallery)
% performs random forest ensemble testing:
% 
%   'model'   The learned ensemble model for classification.
%   'probe'   The DxP data matrix containing the probe samples, one sample
%             per column.
%   'gallery' The DxQ data matrix containing the gallery samples, one
%             sample per column.
%   'scores'  The PxQ matching score matrix where scores(p,q) is the
%             matching score between the j-th sample from 'probe' and the
%             q-th sample from 'gallery'. The scores are bewteen 0 and 1,
%             the higher matching score indicates better matching.
%
% See also PRMI_RANDOM_FOREST_TRAIN.
%
% Contact: www.dihong.me

if size(probe,1) ~= size(gallery,1)
    error('Dimensions of probe and gallery are inconsistent.');
end
validateattributes(probe,{'double'},{'nonsparse','finite','nonnan'},mfilename,'probe',2);
validateattributes(gallery,{'double'},{'nonsparse','finite','nonnan'},mfilename,'gallery',3);
scores = 0;
for i = 1:length(model.models)
    scores = scores + feval(model.cfTest,model.models{i},probe(model.attributes{i},:),gallery(model.attributes{i},:));
end
scores = (1+scores)/2;
end