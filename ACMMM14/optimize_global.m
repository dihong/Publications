%% This function [globally] optimizes the model parameters for WGP for regression.
% It is a generalized WGP which includes group concept.
% #Inputs:
% [x] dxn training data matrix.
% [y] 1xn training target.
% #Outputs:
% [theta] a 1x2 kernel parameter vector, where K(i,j) = theta(1)^2*exp(-|x[i,k]-x[j,k]|^2/(2*theta(2)^2))
% [sigma] a scalar for value of noise level.
% [phi] 1x4 parameters [a,b,c,d] for warp function: a*log(b*x+c)+d.
function [theta,sigma,phi] = optimize_global(x,y)
step = 1e-3; % line search step.
MaxEpoch = 20; %maximum epoch for global iterations.
E = zeros(1,MaxEpoch);
phi = sqrt([1 1/mean(y) 0.01 0]);  % warp parameters: g(y) = phi[1]^2*log(phi[2]^2*y+phi[3]^2) +phi[4]
% phi = [2 sqrt(1/mean(y)) -1 0];
sigma = 0.8; % estimation for the std.
Dx = compute_pairwise_distance(x);
theta = [1 sqrt(mean(Dx(:)))/2];
I = eye(numel(y));
for it = 1:MaxEpoch
    [g dy dg dydg] = warp_func(y,phi); %warp targets.
    % optimize [sigma]
    K = theta(1)^2*exp(-Dx/(2*theta(2)^2)); %kernel
    sigma2 = sigma^2; %squared sigma.
    Kt = K + sigma2*I;
    L = chol(Kt);
    invKtg = L\(L'\g');
    invKt = L\(L'\I);
    invKtdg = L\(L'\dg');
    KD = invKt-invKtg*invKtg';
    ds = sigma*trace(KD); %derivative w.r.t. sigma.
    dt(1) = sum(sum(KD.*K))/theta(1); %derivative w.r.t. theta(1).
    dt(2) = sum(sum(KD.*K.*Dx))/(theta(2)^3); %derivative w.r.t. theta(2).
    dp = g*invKtdg-sum(dydg,2)';%derivative w.r.t. phi.
    nlk_0 = (g*invKtg+2*sum(log(diag(L))))/2-sum(log(dy)); %current performance.
    
    while 1 %line search.
        sigma_1 = sigma - step*ds;
        theta_1 = theta - step*dt;
        phi_1 = phi - step*dp;
        [g dy] = warp_func(y,phi_1); %warp targets.
        K = theta_1(1)^2*exp(-Dx/(2*theta_1(2)^2)); %kernel
        sigma2 = sigma_1^2; %squared sigma.
        Kt = K + sigma2*I;
        L = chol(Kt);
        invKtg = L\(L'\g');
        nlk_1 = (g*invKtg+2*sum(log(diag(L))))/2-sum(log(dy)); %new performance.
        if nlk_1<nlk_0
            sigma = sigma_1;
            theta = theta_1;
            phi = phi_1;
            step = step*2;
            break;
        else
            step = step/5;
            if step<1e-20
                break;
            end
        end
    end
    if step<1e-20
        break;
    end
    % compute current nlk.
    E(it) = nlk_1; %current performance.
    if it>2 && E(it)>E(it-2)*(1-1e-3)
        break;
    end
%     S(it) = step;
end
% nlk = E(it);
sigma = abs(sigma);
end
