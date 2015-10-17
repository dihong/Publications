function [S] = minMaxNorm(S)
S = bsxfun(@rdivide,bsxfun(@minus,S,min(S,[],2)),(max(S,[],2)-min(S,[],2)));
end