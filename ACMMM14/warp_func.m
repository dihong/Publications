function [g dy dg dydg] = warp_func(Y,phi,xmin,xmax)
if nargin==2
    if nargout==1
        g = warp(Y,phi);
    elseif nargout==2
        [g dy] = warp(Y,phi);
    else
        [g dy dg dydg] = warp(Y,phi);
    end
else
    g = iwarp(Y,phi,xmin,xmax);
end
end



%% This function return the inverse warp function.
% The solution must be within xmin and xmax. If no such solution exists,
% return xmin or max, whichever close to the true solution.

function X = iwarp(Y,phi,xmin,xmax)
K = 5000;
xref = linspace(xmin,xmax,K);
yref = warp(xref,phi);
%binary search.
X = zeros(1,length(Y));
for n = 1:length(Y) %for each probe sample.
    xL = 1;
    xR = length(xref);
    y = Y(n);
    while xL<=xR
        mid = round((xL+xR)/2);
        if y > yref(mid)
            xL = mid+1;
        else
            xR = mid-1;
        end
    end
    if mid > K
        mid = K;
    elseif mid < 1;
        mid = 1;
    end
    X(n) = xref(mid);
end
end
% function X = iwarp(Y,phi,xmin,xmax)
% X = Y;
% end

%%
% function [g dy dg dydg] = warp(y,phi)
% g = y;
% dy = ones(size(y));
% 
% dg = zeros(4,numel(y));
% dydg = zeros(4,numel(y));
% end

function [g dy dg dydg] = warp(y,phi)
a = phi(1);
b = phi(2);
c = phi(3);
d = phi(4);
t1 = b^2*y+c^2;

g = a^2*log(t1)+d;

if nargout==1
    return;
end

dy = a^2*b^2./t1;

if nargout==2
    return;
end

dg = zeros(4,numel(y));
dydg = zeros(4,numel(y));
dg(1,:) = 2*a*log(t1); %squared sigma.
dg(2,:) = 2*a^2*b*y./t1;
dg(3,:) = 2*a^2*c./t1;
dg(4,:) = ones(1,numel(y));

dydg(1,:) = 2/a*ones(1,numel(y));
dydg(2,:) = 2/b-2*b*y./t1;
dydg(3,:) = -2*c./t1;
dydg(4,:) = zeros(1,numel(y));

end