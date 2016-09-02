function scores = prmi_usa_test(model,probe,gallery)
% PRMI_USA_TEST Universal Subspace Analysis.
% scores = PRMI_USA_TEST(model,probe,gallery) performs USA testing:
%   'model'   The trained USA model for classification.
%   'probe'   The DxP data matrix containing the probe samples, one sample
%             per column.
%   'gallery' The DxQ data matrix containing the gallery samples, one
%             sample per column.
%   'scores'  The PxQ matching score matrix where scores(p,q) is the
%             matching score between the j-th sample from 'probe' and the
%             q-th sample from 'gallery'. The scores are bewteen 0 and 1,
%             the higher matching score indicates better matching.
%
% See also PRMI_USA_TRAIN, PRMI_MCA_TEST.
%
% Contact: www.dihong.me

if size(probe,1) ~= size(gallery,1)
    error('Dimensions of probe and gallery are inconsistent.');
end

validateattributes(model.P,{'double','single'},{'nonsparse','finite','nonnan','real'},mfilename,'model.P',1);
validateattributes(model.M,{'double','single'},{'nonsparse','finite','nonnan','real'},mfilename,'model.M',1);
validateattributes(probe,{'double','single'},{'nonsparse','finite','nonnan','real'},mfilename,'probe',2);
validateattributes(gallery,{'double','single'},{'nonsparse','finite','nonnan','real'},mfilename,'gallery',3);

scores = prmi_l2_norm(model.P'*bsxfun(@minus,probe,model.M))'*prmi_l2_norm(model.P'*bsxfun(@minus,gallery,model.M));
scores = (1+scores)/2;
end