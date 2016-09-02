function scores = prmi_matching_scores(gallery,probe,type)
% PRMI_MATCHING_SCORES Compute matching scores.
% scores = PRMI_MATCHING_SCORES(gallery,probe) computes the matching scores
% between probe and gallery features.
%
%   'gallery'  DxP matrix with one feature per column for gallery features.
%   'probe'    DxQ matrix with one feature per column for probe features.
%   'scores'   QxP matrix where scores(q,p) is the matching score between
%              the q-th probe feature and the p-th gallery feature.
%   'type'     [Optional] A string specifies the type of matching scores. 
%              The type can be either 'cosine' or 'euclidean', representing
%              cosine distance and euclidean distance respectively. Default
%              is cosine. 
%
% See also PRMI_COMPUTE_TOP
%
% Contact: www.dihong.me

if nargin<3
    type='cosine';
end

if strcmp(type,'cosine')
    scores = prmi_l2_norm(probe)'*prmi_l2_norm(gallery);
elseif strcmp(type,'euclidean')
    scores = 1 - prmi_compute_pairwise_distance(gallery,probe);
else
    error(['prmi_matching_scores: unknown type ' type '.']);
end
end