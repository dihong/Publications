%% This function optimizes the model parameters for WGP for regression.
% It is a generalized WGP which includes group concept.
% #Inputs:
% [x] dxn training data matrix.
% [y] 1xn training target.
% #Outputs:
% [theta] a 1x3 kernel parameter vector, where K(i,j) =
% theta(1)^2*exp(-sum(beta(k)^2*(|x[i,k]-x[j,k]|^2)))+theta(2)^2+theta(3)^2*x[i]'*x[j]
% [beta] 1xD kernel parameters, where D is the dimension of feature x.
% [sigma] a scalar for initial value of noise level.
% [phi] 1x4 parameters [a,b,c,d] for warp function: a*log(b*x+c)+d.
function [theta,beta,sigma,phi] = optimize(x,y)
s1 = 1e-3; %step for [sigma]
s2 = 1e-2; %step for [theta] and [beta]
s3 = 1e-3; %step for [phi]
MaxEpoch = 10; %maximum epoch for global iterations.
[dim N] = size(x);
beta = 0.5*ones(dim,1); % K(i,j) = theta(1)^2*exp(-sum(beta(k)^2*(|x[i,k]-x[j,k]|^2)))+theta(2)^2+theta(3)^2*x[i]'*x[j].   *(1/sqrt(mean(mean(compute_pairwise_distance(x))))*2)
E = zeros(1,MaxEpoch);
phi = sqrt([1 1/mean(y) 1 0]);  % warp parameters: g(y) = phi[1]^2*log(phi[2]^2*y+phi[3]^2) +phi[4]
sigma = 0.5; % estimation for the std.
theta = sqrt([1 0 0]);
I = sparse(eye(N));
XtX = x'*x;
for it = 1:MaxEpoch
    [Dx Cx]= compute_pairwise_distance(x,x,beta.*beta); %pairwise weighted squared Euclidean distances.
    [g dg dydg dy] = warp(y,phi); %warp targets.
    % optimize [sigma]
    K = theta(1)^2*exp(-Dx/2)+theta(2)^2+theta(3)^2*Cx; %kernel
    sigma2 = sigma^2; %squared sigma.
    Kt = K + sigma2*I;
    L = chol(Kt);
    invKtg = L\(L'\g');
    invKt = L\(L'\I);
    delta = diag(invKt-invKtg*invKtg');
    ds = sigma*sum(delta);
    sigma_0 = sigma;
    nlk_0 = (g*invKtg+2*sum(log(diag(L))))/2+sum(log(sigma2)); %current performance.
    while 1 %line search.
        sigma_1 = sigma_0 - s1*ds;
        sigma2 = sigma_1.^2;
        Kt = K + sigma2*I;
        L = chol(Kt);
        invKtg = L\(L'\g');
        nlk_1 = (g*invKtg+2*sum(log(diag(L))))/2+sum(log(sigma2)); %current performance.
        if nlk_1 < nlk_0  %accept this update.
            s1 = s1 * 2;
            sigma = sigma_1;
            break;
        else %try another step.
            s1 = s1 / 5;
            if s1 < 1e-20 %converged.
                break;
            end
        end
    end
    
    % optimize [theta,beta]
    sigma2 = sigma^2; %squared sigma.
    Kt = K + sigma2*I;
    L = chol(Kt);
    invKt = L\(L'\I);
    invKtg = L\(L'\g');
    KD = invKt-invKtg*invKtg';
    tmp = exp(-Dx/2);
    dt(1) = theta(1)*sum(sum(KD.*tmp)); %derivative w.r.t. theta.
    dt(2) = theta(2)*sum(sum(KD));
    dt(3) = theta(3)*sum(sum(KD.*XtX));
    
    
%     db = -beta*theta(1)^2*sum(sum(KD.*tmp.*compute_pairwise_distance(x)));
    tmp = KD.*tmp*(-theta(1)^2/2);
    db = beta_derivative(tmp,x); %derivative w.r.t. beta.
    
    theta_0 = theta;
    beta_0 = beta;
    nlk_0 = (g*invKtg+2*sum(log(diag(L))))/2; %current performance.
    while 1 %line search.
        theta_1 = theta_0 - s2*dt;
        beta_1 = beta_0 - s2*db;
        [Dx Cx]= compute_pairwise_distance(x,x,beta_1.*beta_1); %pairwise weighted squared Euclidean distances.
        K = theta_1(1)^2*exp(-Dx/2)+theta_1(2)^2+theta_1(3)^2*Cx; %kernel
        Kt = K + sigma2*I;
        L = chol(Kt);
        invKtg = L\(L'\g');
        nlk_1 = (g*invKtg+2*sum(log(diag(L))))/2; %current performance.
        if nlk_1 < nlk_0  %accept this update.
            s2 = s2 * 2;
            theta = theta_1;
            beta = beta_1;
            break;
        else
            s2 = s2 / 5;
            if s2 < 1e-20
                break;
            end
        end
    end
    
    % optimize phi
    invKtdg = L\(L'\dg');
    dp = g*invKtdg-sum(dydg,2)';
    nlk_0 = g*invKtg/2-sum(log(dy)); %current performance.
    phi_0 = phi;
    while 1
        phi_1 = phi_0 - s3*dp;
        [g dg dydg dy] = warp(y,phi_1);
        invKtg = L\(L'\g');
        nlk = g*invKtg/2-sum(log(dy)); %current performance.
        if nlk < nlk_0  %accept this update.
            s3 = s3 * 2;
            phi = phi_1;
            break;
        else
            s3 = s3 / 5;
            if s3 < 1e-20
                break;
            end
        end
    end
    
    % compute current nlk.
    E(it) = (g*invKtg+2*sum(log(diag(L))))/2-sum(log(dy))+sum(log(sigma2)); %current performance.
    if it>2 && E(it)>E(it-2)*(1-1e-2)
        break;
    end
    S(it) = s1;
end
% nlk = E(it);
end

function [g dg dydg dy] = warp(y,phi)
a = phi(1);
b = phi(2);
c = phi(3);
d = phi(4);
t1 = b^2*y+c^2;

g = a^2*log(t1)+d;

if nargout==1
    return;
end

dg = zeros(4,numel(y));
dydg = zeros(4,numel(y));
dg(1,:) = 2*a*log(t1);a.^2; %squared sigma.
dg(2,:) = 2*a^2*b*y./t1;
dg(3,:) = 2*a^2*c./t1;
dg(4,:) = ones(1,numel(y));

dydg(1,:) = 2/a*ones(1,numel(y));
dydg(2,:) = 2/b-2*b*y./t1;
dydg(3,:) = -2*c./t1;
dydg(4,:) = zeros(1,numel(y));

dy = a^2*b^2./t1;
end
