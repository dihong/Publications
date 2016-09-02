function [Ft,F] = prmi_dsift(varargin)
% PRMI_DSIFT Dense SIFT (DSIFT) feature extraction.
% [Ft,F] = PRMI_DSIFT(I) performs DSIFT feature extraction on 2-D image I.
% Feature Ft is Dx1 dimensional feature, F is dxn feature matrix where each
% column contains one local feature.
%
% [Ft,F] = PRMI_DSIFT(I,OPTION1,OPTION2,...) performs DSIFT feature extraction
% with specified options. Option arguments can have the following values:
%
%   'winsize'  The size of window based on which histogram is calculated.
%              Default value is 12. The winsize must be even number.
%
%   'step'     The distance between two neighboring landmarks. If 'landmarks'
%              option is specified, the specification of 'step' will be
%              ignored. Default value is 8.
%
%   'nbins'  The number of spatial bins for histogram calculation.
%              Default value is 12. 
%
% See also PRMI_MLBP, PRMI_HOG, PRMI_BIF
%
% Contact: www.dihong.me

if length(varargin)<1
    error('prmi_dsift: At least one argument required.');
else
    I = varargin{1};
end

validateattributes(I,{'uint8'},{'nonsparse','finite','nonnan','2d'},mfilename,'I',1);

allStrings = {'winsize', 'step', 'nbins'};

%defaults
winsize = 12;
step = 8;
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
        case {'nbins'}
            validateattributes(varargin{k+1},{'double'}, {'positive'},mfilename,'nbins',1+k);
            nbins = varargin{k+1};
    end
  else
      error('prmi_dsift: incorrect option format.');
  end
end

%run the dsift.
F = DSIFT(I,winsize,step,nbins);
Ft = F(:);
end
