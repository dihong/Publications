%% This function [globally] optimizes the model parameters for WGP for regression.
% It is a generalized WGP which includes group concept.
% #Inputs:
% [x] dxn training data matrix.
% [y] 1xn training target.
% #Outputs:
% [theta] a 1x2 kernel parameter vector, where K(i,j) = theta(1)^2*exp(-|x[i,k]-x[j,k]|^2/(2*theta(2)^2))
% [sigma] a scalar for value of noise level.
% [phi] 1x4 parameters [a,b,c,d] for warp function: a*log(b*x+c)+d.
function [theta,sigma,phi] = optimize_global_svd(x,y)
MaxEpoch = 10; %maximum epoch for global iterations.
% E = zeros(1,MaxEpoch);
phi = sqrt([1 1/mean(y) 0.01 0.0]);  % warp parameters: g(y) = phi[1]^2*log(phi[2]^2*y+phi[3]^2) +phi[4]
% phi = [2 sqrt(1/mean(y)) -1 0];
sigma = ones(numel(y),1); % estimation for the std.
Dx = compute_pairwise_distance(x);
theta = [1 sqrt(mean(Dx(:)))];
A = exp(-Dx/(2*(theta(2)^2))); %similarity matrix.
[U S V] = svd(A); 
b1 = diag(S);
b0 = ones(numel(b1),1);


for it = 1:MaxEpoch
    lamda = sigma.^2+theta(1)^2*b1; %eigen values.
    [g dy dg dydg] = warp_func(y,phi); %warp targets.
    q = U'*g';
    q2 = q.*q;
    
    % optimize phi.
    step = 1e-3;
    while step>1e-10
        invKtdg = dg*U*bsxfun(@times,U',1./lamda);
        dp = g*invKtdg'-sum(dydg,2)';%derivative w.r.t. phi.
        perf1 = sum(log(lamda)) + (1./lamda)'*q2-sum(log(dy));
        while 1
            p1 = phi - step*dp;
            [g dy dg dydg] = warp_func(y,p1); %warp targets.
            q = U'*g';
            q2 = q.*q;
            perf2 = sum(log(lamda)) + (1./lamda)'*q2-sum(log(dy));
            if perf2 < perf1
                phi = p1;
                step = step*2;
                break;
            else
                step = step / 5;
                if step <= 1e-10
                    break;
                end
            end
        end
    end
    [g dy] = warp_func(y,phi); %warp targets.
    
    % optimize [sigma,theta]
    q = U'*g';
    q2 = q.*q;
    step = 1e-3;
    nc = 5;
    [index values] = kmeans(q2-theta(1)^2*b1,nc,'EmptyAction','drop');
    while step>1e-10
        lamda = sigma.^2+theta(1)^2*b1; %eigen values.
        t = (lamda-q2)./(lamda.^2);
        ds = zeros(numel(y),1);
        for i = 1:nc
            ds(index==i) = sigma(index==i)'*t(index==i);
        end
        dt = theta(1)*t'*b1; %derivative w.r.t. theta(1).
        perf1 = sum(log(lamda)) + (1./lamda)'*q2-sum(log(dy));
        while 1
            t1 = theta(1) - step*dt;
            s1 = sigma - step*ds;
            lamda = s1.^2+t1^2*b1; %eigen values.
            perf2 = sum(log(lamda)) + (1./lamda)'*q2-sum(log(dy));
            if perf2 < perf1*.99
                theta(1) = t1;
                sigma = s1;
                step = step*2;
                break;
            else
                step = step / 5;
                if step <= 1e-10
                    break;
                end
            end
        end
    end
%     
%     tmp = [b0 b1]\q2;
%     
%     sigma = sqrt(tmp(2))*ones(size(sigma));
%     
%     theta(1) = sqrt(tmp(1));
%     
%     lamda = sigma.^2+theta(1)^2*b1; %eigen values.
% 
%     perf2 = sum(log(lamda)) + (1./lamda)'*q2-sum(log(dy));

    E(it) = perf2; %current performance.
    if it>1 && E(it)>E(it-1)*(1-1e-3)
        break;
    end
end


end







% for it = 1:MaxEpoch
%     lamda = sigma.^2+theta(1)^2*b1; %eigen values.
%     [g dy dg dydg] = warp_func(y,phi); %warp targets.
%     q = U'*g';
%     q2 = q.*q;
%     
%     % optimize phi.
%     step = 1e-3;
%     while step>1e-10
%         invKtdg = dg*U*bsxfun(@times,U',1./lamda);
%         dp = g*invKtdg'-sum(dydg,2)';%derivative w.r.t. phi.
%         perf1 = sum(log(lamda)) + (1./lamda)'*q2-sum(log(dy));
%         while 1
%             p1 = phi - step*dp;
%             [g dy dg dydg] = warp_func(y,p1); %warp targets.
%             q = U'*g';
%             q2 = q.*q;
%             perf2 = sum(log(lamda)) + (1./lamda)'*q2-sum(log(dy));
%             if perf2 < perf1
%                 phi = p1;
%                 step = step*5;
%                 break;
%             else
%                 step = step / 10;
%                 if step <= 1e-10
%                     break;
%                 end
%             end
%         end
%     end
%     [g dy] = warp_func(y,phi); %warp targets.
%     
%     % optimize [sigma,theta]
%     q = U'*g';
%     q2 = q.*q;
%     
%     tmp = [b0 b1]\q2;
%     
%     sigma = sqrt(tmp(1));
%     
%     theta(1) = sqrt(tmp(2));
% 
%     lamda = sigma.^2+theta(1)^2*b1; %eigen values.
%     perf2 = sum(log(lamda)) + (1./lamda)'*q2-sum(log(dy));
%     E(it) = perf2; %current performance.
%     if it>1 && E(it)>E(it-1)*(1-1e-3)
%         break;
%     end
% end
















