function avgTop=prmi_compute_top(scores,top)
% PRMI_COMPUTE_TOP computes the recognition accuracy.
% avgTop = PRMI_COMPUTE_TOP(scores,top) computes the recognition accuracy of rank top.
%
%   'scores'   QxP matrix where scores(q,p) is the matching score between
%              the q-th probe feature and the p-th gallery feature.
%   'top'      [Optional] A positive integer specifies the rank. Default 
%              value is 1.
%
% See also PRMI_MATCHING_SCORES
%
% Contact: www.dihong.me

if nargin == 1 
    top = 1;
end
score=zeros(top,size(scores,1));
for i = 1:size(scores,1)
    [~,index] = sort(scores(i,:),'ascend');
    for k2 = 1:top
        score(k2,i) = score(k2,i) + ~isempty(find(index((end-k2+1):end) ==i,1));
    end
end
avgTop=sum(score,2)/size(score,2);
end
