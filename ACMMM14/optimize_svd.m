%% This function [locally] optimizes the model parameters for WGP for regression.
% It is a generalized WGP which includes group concept.
% #Inputs:
% [x] dxn training data matrix.
% [y] 1xn training target.
% #Outputs:
% [theta] a 1x2 kernel parameter vector, where K(i,j) = theta(1)^2*exp(-|x[i,k]-x[j,k]|^2/(2*theta(2)^2))
% [sigma] a scalar for value of noise level.
% [phi] 1x4 parameters [a,b,c,d] for warp function: a*log(b*x+c)+d.
function [theta,w,phi,rho] = optimize_svd(x,y)
Dx = compute_pairwise_distance(x);
phi = sqrt([1 1/mean(y) 0.01 0]);  % warp parameters: g(y) = phi[1]^2*log(phi[2]^2*y+phi[3]^2) +phi[4]
rho = 2/mean(Dx(:));
A = exp(-rho*Dx); %similarity matrix.
[U S V] = svd(A);
b1 = diag(S);
b0 = ones(numel(b1),1);
nb0 = norm(b0);
nb1 = norm(b1);
B = orth([b0 b1]);
[g dy dg dydg] = warp_func(y,phi); %warp targets.


q = U'*g';
q2 = q.*q;
q2 = B*B'*q2;
d1 = b0'*q2/nb0;
d2 = sqrt(q2'*q2-d1^2);
d5 = d2/(sin(acos(b0'*b1/(nb0*nb1))));
d3 = sqrt(d5^2-d2^2);
d4 = d1 - d3;
d5 = d3/(b0'*b1/(nb0*nb1));
theta = d5/nb1;
sigma = max(0,d4/nb0);

w = U*(diag(1./(theta*b1+sigma))*(U'*g'));

return;


s3 = 1e-4; % line search step.
MaxEpoch = 10; %maximum epoch for global iterations.
for it = 1:MaxEpoch
    % optimize [phi]
    invKtg = U*(diag(1./(theta*b1+sigma))*(U'*g'));
    invKtdg = U*(diag(1./(theta*b1+sigma))*(U'*dg'));
    dp = g*invKtdg-sum(dydg,2)';%derivative w.r.t. phi.
    nlk_0 = g*invKtg/2-sum(log(dy)); %current performance.
    while s3>1e-20 %line search.
        phi_1 = phi - s3*dp;
        [g dy dg dydg] = warp_func(y,phi_1); %warp targets.
        invKtg = U*(diag(1./(theta*b1+sigma))*(U'*g'));
        nlk_1 = g*invKtg/2-sum(log(dy));
        if nlk_1<nlk_0
            phi = phi_1;
            s3 = s3*2;
            break;
        else
            s3 = s3/5;
        end
    end
    
    if s3<=1e-20
        break;
    end
    
    % optimize [sigma], [theta]
    q = U'*g';
    q2 = q.*q;
    q2 = B*B'*q2;
    d1 = b0'*q2/nb0;
    d2 = sqrt(q2'*q2-d1^2);
    d5 = d2/(sin(acos(b0'*b1/(nb0*nb1))));
    d3 = sqrt(d5^2-d2^2);
    d4 = d1 - d3;
    d5 = d3/(b0'*b1/(nb0*nb1));
    theta = d5/nb1;
    sigma = max(0,d4/nb0);
    t(it) = sigma;
end
w = U*(diag(1./(theta*b1+sigma))*(U'*g'));
% nlk = E(it);
end

