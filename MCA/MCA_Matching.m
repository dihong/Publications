%% This function implements face matching based on MCA model.
% [probe] DxP data matrix consists of P column observations.
% [gallery] DxQ data matrix consists of Q column observations.
% [model] trained MCA model at the training stage.
% return: [score] is a PxQ matrix contains matching scores. score(i,j) is
% the matching score between i-th probe feature and j-th gallery feature.
function score = MCA_Matching(probe,gallery,model)
    [gallery Cg]=computeEy(gallery,model,1); %gallery.
    [probe Cp]=computeEy(probe,model,2); %probe.
    gallery = bsxfun(@rdivide,gallery,sqrt(sum(gallery.^2,1)));
    probe = bsxfun(@rdivide,probe,sqrt(sum(probe.^2,1)));
    invCg = inv(Cg);
    invCp = inv(Cp);
    invA = inv(invCg + invCp);
    Vg = invCg*gallery;
    W = invCp*invA*Vg;
    W0 = diag(Vg'*invA*Vg - gallery'*invCg*gallery)'/2; %row vector.
    score = bsxfun(@plus,W0,probe'*W);
end
