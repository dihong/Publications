function [ProjectMatrix Mean_Vector]= DiscriminantAnalysis(X,PCAN,IN,BN)

%    X is the data matrix for training ;
%    N is the number of samples for each individual;
%    PCAN is the dimensionality of the first reduced space ;
%    IN is the number of the second reduced space;
%    BN is the number of the final subspace.
%
N=2;
[FeatureLength SampleNumber]=size(X);
ClassNum=round(SampleNumber/N);

if (PCAN ~= 0)
	[eigenvectors, Mean_Vector]=PCA(X);
	Select_eigenvectors=eigenvectors(:,1:PCAN);
	W=Select_eigenvectors'*(X-repmat(Mean_Vector,1,SampleNumber));
	ClassCenters(1:PCAN,ClassNum)=0;
	for i=1:N
	   ClassCenters=ClassCenters+W(:,i:N:end);
	end
   ClassCenters=ClassCenters./N;
else
   W = X;
   ClassCenters(1:FeatureLength,ClassNum)=0;
	for i=1:N
	   ClassCenters=ClassCenters+W(:,i:N:end);
	end
   ClassCenters=ClassCenters./N;
end

for i=1:N
   W(:,i:N:end)=W(:,i:N:end)-ClassCenters;   
end

[W_val, W_vec, W_m]=PCA2(W);
if (IN==0)
   SW_val=W_val;
   SW_vec=W_vec;
else
   SW_val=W_val(1:IN);
   SW_vec=W_vec(:,1:IN);
end

if (PCAN~=0)
    SW_vec=SW_vec./(repmat(SW_val',[PCAN,1]).^0.5);
else
    SW_vec=SW_vec./(repmat(SW_val',[FeatureLength,1]).^0.5);
end
m=mean(ClassCenters,2);
M=repmat(m,1,ClassNum);
B=SW_vec'*(ClassCenters-M);
Between_Class_Matrix=B*B';
if ((BN==0)&(PCAN~=0))
   ProjectMatrix=Select_eigenvectors*SW_vec;
elseif ((BN==0)&(PCAN==0))
   ProjectMatrix=SW_vec; 
elseif ((BN~=0)&(PCAN==0))
   [B_val,B_vec,B_m]=PCA2(B);
   SB_vec=B_vec(:,1:BN);
   ProjectMatrix=SW_vec*SB_vec;
else
   [B_val,B_vec,B_m]=PCA2(B);
   SB_vec=B_vec(:,1:BN);
   ProjectMatrix=Select_eigenvectors*SW_vec*SB_vec;
end

