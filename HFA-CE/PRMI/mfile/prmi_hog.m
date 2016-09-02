function [Ft,F,P] = prmi_hog(varargin)
% PRMI_HOG Histogram of Oriented Gradients (HOG) feature extraction.
% [Ft,F,P] = PRMI_HOG(I) performs HOG feature extraction on 2-D image I. 
% Feature Ft is Dx1 dimensional feature, F is dxn feature matrix where each
% column of F is extracted local HOG feature at the corresponding column of
% P which specifies the landmark location of that HOG feature.
%
% [Ft,F,P] = PRMI_HOG(I,OPTION1,OPTION2,...) performs HOG feature extraction
% with specified options. Option arguments can have the following values:
%
%   'winsize'  The size of window based on which histogram is calculated.
%              Default value is 16. The winsize must be even number.
%
%   'step'     The distance between two neighboring landmarks. If 'landmarks'
%              option is specified, the specification of 'step' will be
%              ignored. Default value is 8.
%
%   'landmarks'  A matrix specifying the location of landmarks with each
%              row containing a coordinate pair (x,y) where x is the row
%              location and y is the column location. By default, the
%              landmark is calculated by 'step' option.
%
%   'nbins'  The number of spatial bins for histogram calculation.
%              Default value is 12. 
%
% See also PRMI_MLBP, PRMI_DSIFT, PRMI_BIF
%
% Contact: www.dihong.me

if length(varargin)<1
    error('prmi_hog: At least one argument required.');
else
    I = varargin{1};
end

validateattributes(I,{'uint8'},{'nonsparse','finite','2d'},mfilename,'I',1);

allStrings = {'winsize', 'step', 'landmarks', 'nbins'};

%defaults
winsize = 16;
step = 8;
landmarks = [];
nbins = 12;

%parse inputs.
for k = 2:2:length(varargin)
  if ischar(varargin{k})
    string = lower(validatestring(varargin{k}, allStrings, mfilename, 'OPTION',k));
    switch string
        case {'winsize'}
            validateattributes(varargin{k+1},{'double'}, {'positive','even'},mfilename,'winsize',1+k);
            winsize = varargin{k+1};
        case {'step'}
            validateattributes(varargin{k+1},{'double'}, {'positive'},mfilename,'step',1+k);
            step = varargin{k+1};
        case {'landmarks'}
            validateattributes(varargin{k+1},{'double'}, {'positive','nrows',2},mfilename,'landmarks',1+k);
            landmarks = varargin{k+1};
        case {'nbins'}
            validateattributes(varargin{k+1},{'double'}, {'positive'},mfilename,'nbins',1+k);
            nbins = varargin{k+1};
    end
  else
      error('prmi_hog: incorrect option format.');
  end
end

%run the HOG.
if nargout==3
    [F,P] = HOG(I,winsize,step,int32(landmarks'),nbins);
else
    F = HOG(I,winsize,step,int32(landmarks'),nbins);
end
Ft = F(:);
end
