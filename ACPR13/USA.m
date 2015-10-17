function [ProjectionMat m]=USA(t,d1,d2,d3)
t=L2Norm(t);
[t m]=meanNorm(t);
ProjectionMat = DiscriminantAnalysis(t,d1,d2,d3);
end