function Y = prmi_slice(X,N)
% PRMI_SLICE slices high dimensional data into several low dimensional parts.
% Y = slicing(X,N) slices X into N parts. The returned value Y is 1-by-N
% cell array with each element containing one part of X. For example:
% X = [1 2 4
%      4 2 4
%      1 1 8
%      2 1 9]
% Then Y = slice(X,2) will return:
% Y{1} = [1 2 4      and    Y{2} = [1 1 8
%         4 2 4]                    2 1 9]
%
% Contact: www.dihong.me

if N>size(X,1)
    error('N must be less than or equal to size(X,1).');
end
d = round(size(X,1)/N);
Y = cell(1,N);
for i = 1:N-1
    Y{i} = X(1+(i-1)*d:i*d,:);
end
Y{N} = X(1+(N-1)*d:end,:);
end