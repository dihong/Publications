function feat = prmi_bif(varargin)
% PRMI_BIF Bio-Inspired Features (BIF).
% Ft = PRMI_BIF(I) performs BIF feature extraction on 2-D image I.
% Feature Ft is Dx1 dimensional feature.
%
% Ft = PRMI_BIF(I,OPTION1,OPTION2,...) performs BIF feature extraction
% with specified options. Option arguments can have the following values:
%
%   'nTheta'   The number of different orientations, such as 4, 6, 8, 10, 12.
%              Default value is 12.
%
%   'PoolWinSize'   A 1xK array of integers where PoolWinSize(k) is the
%               size of pooling window for the k-th band. There are K bands
%               in total. Default value is 6:3:21.
%
%   'FS1'      A 1xK array of integers where FS1(k) is the smaller window size of
%              the Gabor filters. There are K total filters. Default value
%              is 4:2:14.
%
%   'FS2'      A 1xK array of integers where FS2(k) is the larger window size of
%              the Gabor filters. There are K total filters. Default value
%              is 6:2:16.
%
% See also PRMI_DSIFT, PRMI_HOG, PRMI_MLBP
%
% Contact: www.dihong.me

if length(varargin)<1
    error('prmi_bif: At least one argument required.');
else
    I = varargin{1};
end

validateattributes(I,{'uint8'},{'nonsparse','finite','2d'},mfilename,'I',1);

allStrings = {'nTheta', 'PoolWinSize', 'FS1', 'FS2'};


%defaults
nTheta = 12;
PoolWinSize = 6:3:21;
FS1 = 4:2:14;
FS2 = 6:2:16;

%parse inputs.
for k = 2:2:length(varargin)
  if ischar(varargin{k})
    string = validatestring(varargin{k}, allStrings, mfilename, 'OPTION',k);
    switch string
        case {'nTheta'}
            validateattributes(varargin{k+1},{'double'}, {'positive'},mfilename,'nTheta',1+k);
            nTheta = varargin{k+1};
        case {'PoolWinSize'}
            validateattributes(varargin{k+1},{'double'}, {'positive'},mfilename,'PoolWinSize',1+k);
            PoolWinSize = varargin{k+1};
        case {'FS1'}
            validateattributes(varargin{k+1},{'double'}, {'positive'},mfilename,'FS1',1+k);
            FS1 = varargin{k+1};
        case {'FS2'}
            validateattributes(varargin{k+1},{'double'}, {'positive'},mfilename,'FS2',1+k);
            FS2 = varargin{k+1};
    end
  else
      error('prmi_bif: incorrect option format.');
  end
end

%check arguments consistency.
if numel(PoolWinSize) ~= numel(FS1) || numel(PoolWinSize) ~= numel(FS2)
    error('prmi_bif: dimensions of PoolWinSize, FS1 and FS2 must be the same.');
end


Theta = pi*(0:nTheta-1)/nTheta;
Sigma1 = FS1/2;
Lamda1 = Sigma1 * 1.25;
Sigma2 = FS2/2;
Lamda2 = Sigma2 * 1.25;
nBand = length(FS1);
feat = [];
for i = 1:nBand
    for j = 1:nTheta
        I1 = gaborfilter(I,Sigma1(i),Lamda1(i),FS1(i),Theta(j));
        I2 = gaborfilter(I,Sigma2(i),Lamda2(i),FS2(i),Theta(j));
        Imax = max(I1,I2);
        feat = [feat ;Pool(Imax,PoolWinSize(i),size(Imax))]; % pool the features from the S1 layer.
    end
end

end




function [gabout, G] = gaborfilter(I,sigma,lamda,win_size,theta,G)
if nargin > 5
    I = double(I);
    gabout = imfilter(I,G); 
    return;
end
gama = 0.3;
win_half = floor(win_size/2);
I = double(I);
[y, x]= meshgrid(-win_half:win_half,-win_half:win_half);
xPrime = x*cos(theta) + y*sin(theta);
yPrime = y*cos(theta) - x*sin(theta);
G = exp(-.5*((xPrime/sigma).^2+(gama*yPrime/sigma).^2)).*cos(2*pi*xPrime/lamda);
gabout = imfilter(I,G); 
end
