%% This function [locally] optimizes the model parameters for WGP for regression.
% It is a generalized WGP which includes group concept.
% #Inputs:
% [x] dxn training data matrix.
% [y] 1xn training target.
% #Outputs:
% [theta] a 1x2 kernel parameter vector, where K(i,j) = theta(1)^2*exp(-|x[i,k]-x[j,k]|^2/(2*theta(2)^2))
% [sigma] a scalar for value of noise level.
% [phi] 1x4 parameters [a,b,c,d] for warp function: a*log(b*x+c)+d.
function [theta,sigma,phi] = optimize_local(x,y)
MaxEpoch = 15; %maximum epoch for global iterations.
E = zeros(1,MaxEpoch);
phi = sqrt([1 1/mean(y) 0.01 0]);  % warp parameters: g(y) = phi[1]^2*log(phi[2]^2*y+phi[3]^2) +phi[4]
% phi = [2 sqrt(1/mean(y)) -1 0];
sigma = 0.5; % estimation for the std.
Dx = compute_pairwise_distance(x);
theta = [1 sqrt(mean(Dx(:))/2)];
I = eye(numel(y));
s1 = 1e-3; % line search step.
s2 = 1e-4; % line search step.
s3 = 1e-4; % line search step.
[g dy dg dydg] = warp_func(y,phi); %warp targets.
K = theta(1)^2*exp(-Dx/(2*theta(2)^2)); %kernel
sigma2 = sigma^2; %squared sigma.
Kt = K + sigma2*I;
L = chol(Kt);
invKtg = L\(L'\g');
for it = 1:MaxEpoch
    % optimize [sigma]
    invKtdg = L\(L'\dg');
    invKt = L\(L'\I);
    KD = invKt-invKtg*invKtg';
    ds = sigma*trace(KD); %derivative w.r.t. sigma.
    nlk_0 = (g*invKtg+2*sum(log(diag(L))))/2+sum(log(sigma2)); %current performance.
    while s1 > 1e-20 %line search.
        sigma_1 = sigma - s1*ds;
        sigma2 = sigma_1^2; %squared sigma.
        Kt = K + sigma2*I;
        L = chol(Kt);
        invKtg = L\(L'\g');
        nlk_1 = (g*invKtg+2*sum(log(diag(L))))/2+sum(log(sigma2)); %new performance.
        if nlk_1<nlk_0
            sigma = sigma_1;
            s1 = s1*2;
            break;
        else
            s1 = s1/5;
        end
    end
    
    if s1<=1e-20
        sigma2 = sigma^2;
        Kt = K + sigma2*I;
        L = chol(Kt);
        invKtg = L\(L'\g');
        invKtdg = L\(L'\dg');
    end

    dt(1) = sum(sum(KD.*K))/theta(1); %derivative w.r.t. theta(1).
    dt(2) = sum(sum(KD.*K.*Dx))/(theta(2)^3)/2; %derivative w.r.t. theta(2).
    nlk_0 = (g*invKtg+2*sum(log(diag(L))))/2; %new performance.
    while s2>1e-20 %line search.
        theta_1 = theta - s2*dt;
        K = theta_1(1)^2*exp(-Dx/(2*theta_1(2)^2)); %kernel
        Kt = K + sigma2*I;
        L = chol(Kt);
        invKtg = L\(L'\g');
        nlk_1 = (g*invKtg+2*sum(log(diag(L))))/2; %new performance.
        if nlk_1<nlk_0
            theta = theta_1;
            s2 = s2*2;
            break;
        else
            s2 = s2/5;
        end
    end
    
    if s2<=1e-20
        K = theta(1)^2*exp(-Dx/(2*theta(2)^2)); %kernel
        sigma2 = sigma^2; %squared sigma.
        Kt = K + sigma2*I;
        L = chol(Kt);
        invKtg = L\(L'\g');
        invKtdg = L\(L'\dg');
    end
    
    
    dp = g*invKtdg-sum(dydg,2)';%derivative w.r.t. phi.
    nlk_0 = g*invKtg/2-sum(log(dy)); %current performance.
    while s3>1e-20 %line search.
        phi_1 = phi - s3*dp;
        [g dy dg dydg] = warp_func(y,phi_1); %warp targets.
        invKtg = L\(L'\g');
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
        [g dy dg dydg] = warp_func(y,phi); %warp targets.
        invKtg = L\(L'\g');
    end
    
    E(it) = (g*invKtg+2*sum(log(diag(L))))/2+sum(log(sigma2))-sum(log(dy)); %current performance.
    
    if it>2 && E(it)>E(it-2)*(1-1e-3)
        break;
    end
    S(it) = s3;
end
% nlk = E(it);
end

