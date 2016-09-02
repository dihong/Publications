function scores = prmi_mca_test(model,probe,gallery)
% PRMI_MCA_TEST Mutual Component Analysis.
% scores = PRMI_MCA_TEST(model,probe,gallery) performs MCA testing:
%   'model'   The trained MCA model for classification.
%   'probe'   The DxP data matrix containing the probe samples, one sample
%             per column.
%   'gallery' The DxQ data matrix containing the gallery samples, one
%             sample per column.
%   'scores'  The PxQ matching score matrix where scores(p,q) is the
%             matching score between the j-th sample from 'probe' and the
%             q-th sample from 'gallery'. The scores are bewteen 0 and 1,
%             the higher matching score indicates better matching.
%
% See also PRMI_MCA_TRAIN, PRMI_USA_TEST.
%
% Contact: www.dihong.me

validateattributes(model.beta{1},{'double'},{'nonsparse','finite','nonnan','real'},mfilename,'model.beta{1}',1);
validateattributes(model.U{1},{'double'},{'nonsparse','finite','nonnan','real'},mfilename,'model.U{1}',1);
validateattributes(model.sigma{1},{'double'},{'nonsparse','finite','nonnan','real'},mfilename,'model.sigma{1}',1);
validateattributes(model.beta{2},{'double'},{'nonsparse','finite','nonnan','real'},mfilename,'model.beta{2}',1);
validateattributes(model.U{2},{'double'},{'nonsparse','finite','nonnan','real'},mfilename,'model.U{2}',1);
validateattributes(model.sigma{2},{'double'},{'nonsparse','finite','nonnan','real'},mfilename,'model.sigma{2}',1);
validateattributes(probe,{'double'},{'nonsparse','finite','nonnan','real'},mfilename,'probe',2);
validateattributes(gallery,{'double'},{'nonsparse','finite','nonnan','real'},mfilename,'gallery',3);

[gallery,Cg]=computeEy(gallery,model.beta{1},model.U{1},model.sigma{1});
[probe,Cp]=computeEy(probe,model.beta{2},model.U{2},model.sigma{2});
gallery = bsxfun(@rdivide,gallery,sqrt(sum(gallery.^2,1)));
probe = bsxfun(@rdivide,probe,sqrt(sum(probe.^2,1)));
invCg = inv(Cg);
invCp = inv(Cp);
invA = inv(invCg + invCp);
Vg = invCg*gallery;
W = invCp*invA*Vg;
W0 = diag(Vg'*invA*Vg - gallery'*invCg*gallery)'/2;
scores = bsxfun(@plus,W0,probe'*W);
scores = (scores-min(scores(:)))./(max(scores(:))-min(scores(:)));
end

function [Ey,C]=computeEy(data,beta,U,sigma)
% Estep function is to compute the first and second moment of P(y(s)|Mk(1,2...),V,M)
UC = U'/(sigma*eye(size(U,1))+U*U');
Ey = UC*bsxfun(@minus,data,beta);
C = eye(size(U,2))-UC*U;
end