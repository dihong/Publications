function E = function_to_be_minimized(feat,age,id,s,t1,t2,p2)
parfor k = 1:82
    mae= mtwgp_regression(feat,age,id,k,s,t1,t2,p2);
    E{k} = mae;
end
E = cat(2,E{:});
E = mean(E);
end
% [sigma,mu,rho,theta,phi,nlk] = optimize(x,y,sigma,mu,rho,theta,phi,sg,kg)


function mae = mtwgp_regression(x,y,id,tid,s,t1,t2,p2)
train_x = x(:,id~=tid);
train_y = y(id~=tid);
test_x = x(:,id==tid);
test_y = y(id==tid);
[theta,sigma,phi] = optimize_global(train_x,train_y,s,t1,t2,p2);
m = predict(train_x,train_y,theta,sigma,phi,test_x);
mae = mean(abs(test_y(:)-m(:)));
end
%%
function out = predict(x,y,theta,sigma,phi,test)
Dx = compute_pairwise_distance(x);
I = eye(numel(y));
g = warp_func(y,phi); %warp targets.
% optimize [sigma]
K = theta(1)^2*exp(-Dx/(2*theta(2)^2)); %kernel
sigma2 = sigma^2; %squared sigma.
Kt = K + sigma2*I;
L = chol(Kt);
invKtg = L\(L'\g');

K2 = theta(1)^2*exp(-compute_pairwise_distance(x,test)/(2*theta(2)^2)); %kernel
z = K2'*invKtg; % prediction in the warped space.
out = warp_func(z,phi,0,70); % the prediction in the observation space.
end


%% This function [globally] optimizes the model parameters for WGP for regression.
% It is a generalized WGP which includes group concept.
% #Inputs:
% [x] dxn training data matrix.
% [y] 1xn training target.
% #Outputs:
% [theta] a 1x2 kernel parameter vector, where K(i,j) = theta(1)^2*exp(-|x[i,k]-x[j,k]|^2/(2*theta(2)^2))
% [sigma] a scalar for value of noise level.
% [phi] 1x4 parameters [a,b,c,d] for warp function: a*log(b*x+c)+d.
function [theta,sigma,phi] = optimize_global(x,y,s,t1,t2,p2)
step = 1e-3; % line search step.
MaxEpoch = 20; %maximum epoch for global iterations.
E = zeros(1,MaxEpoch);
phi = sqrt([1 p2/mean(y) 0.01 0]);  % warp parameters: g(y) = phi[1]^2*log(phi[2]^2*y+phi[3]^2) +phi[4]
% phi = [2 sqrt(1/mean(y)) -1 0];
sigma = s; % estimation for the std.
Dx = compute_pairwise_distance(x);
theta = [t1 t2*sqrt(mean(Dx(:)))/2];
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
    nlk_0 = (g*invKtg+2*sum(log(diag(L))))/2-sum(log(dy))+sum(log(sigma2)); %current performance.
    
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
        nlk_1 = (g*invKtg+2*sum(log(diag(L))))/2-sum(log(dy))+sum(log(sigma2)); %new performance.
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
end

