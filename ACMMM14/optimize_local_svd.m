%% This function [globally] optimizes the model parameters for WGP for regression.
% It is a generalized WGP which includes group concept.
% #Inputs:
% [x] dxn training data matrix.
% [y] 1xn training target.
% #Outputs:
% [theta] a 1x2 kernel parameter vector, where K(i,j) = theta(1)^2*exp(-|x[i,k]-x[j,k]|^2/(2*theta(2)^2))
% [sigma] a scalar for value of noise level.
% [phi] 1x4 parameters [a,b,c,d] for warp function: a*log(b*x+c)+d.
function [theta,sigma,phi] = optimize_local_svd(x,y)

MaxEpoch = 10; %maximum epoch for global iterations.
phi = sqrt([1 1/mean(y) 0.01 0.0]);  % warp parameters: g(y) = phi[1]^2*log(phi[2]^2*y+phi[3]^2) +phi[4]
% phi = [2 sqrt(1/mean(y)) -1 0];
sigma = 0.8; % estimation for the std.
Dx = compute_pairwise_distance(x);
theta = [1 sqrt(mean(Dx(:)))/2];
A = exp(-Dx/(2*(theta(2)^2))); %similarity matrix.
[U S V] = svd(A); 
b1 = diag(S);
b0 = ones(numel(b1),1);
B = orth([b0 b1]); % space spanned by (b0,b1).
nb0 = norm(b0);
nb1 = norm(b1);
tmp1 = (sin(acos(b0'*b1/(nb0*nb1))));
tmp2 = (b0'*b1/(nb0*nb1));
for it = 1:MaxEpoch
    lamda = sigma^2+(theta(1)^2)*b1; %eigen values.
    % Fix (sigma,theta), optimize warp.
    step = 1e-3; % line search step.
    tmp3 = U*bsxfun(@times,U',1./lamda);
    while step>1e-20
        [g dy dg dydg] = warp_func(y,phi);
        q = (g*U)';
        q2 = q.*q;
        invKtdg = dg*tmp3;
        dp = g*invKtdg'-sum(dydg,2)';%derivative w.r.t. phi.
        nlk_0 = (1./lamda)'*q2-sum(log(dy)); %current performance.
        while 1 %line search.
            phi_1 = phi - step*dp;
            [g dy] = warp_func(y,phi_1); %warp targets.
            q = (g*U)';
            q2 = q.*q;
            nlk_1 = (1./lamda)'*q2-sum(log(dy)); %current performance.
            if nlk_1<nlk_0
                phi = phi_1;
                step = step*2;
                break;
            else
                step = step/5;
                if step<=1e-20
                    break;
                end
            end
        end
    end
    
    % Fix warp, optimize (theta,sigma).
    g = warp_func(y,phi);
    q = (g*U)';
    q2 = q.*q;
    q2 = B*B'*q2; %project into span(b0,b1)
    d1 = b0'*q2/nb0;
    d2 = sqrt(q2'*q2-d1^2);
    d5 = d2/tmp1;
    d3 = sqrt(d5^2-d2^2);
    d4 = d1 - d3;
    d5 = d3/tmp2;
    sigma = sqrt(max(0,d4/nb0));
    theta(1) = sqrt(d5/nb1);
    
    
    s(it) = sigma;
    t(it) = theta(1);
    a(it) = phi(1);
    b(it) = phi(2);
    c(it) = phi(3);
    d(it) = phi(4);
end
end
