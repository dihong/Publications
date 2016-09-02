function [Ft,F,P,Icode] = prmi_lps(varargin)
% PRMI_LPS_ENCODE Local Patterns Selection (LPS) feature extraction.
% [Ft,F,P,Icode] = PRMI_LPS_ENCODE(I,model) performs LPS feature extraction on 2-D image I.
% Feature Ft is Dx1 dimensional feature. F is a cell array with each element 
% contaning a feature matrix where each column is the local feature extracted 
% at the corresponding column of P which specifies the landmark location of 
% that local LPS feature. The Icode is encoded face image (double). The
% input parameter model is a learned LPS model returned by prmi_lps_train.
%
% [Ft,F,P,Icode] = PRMI_LPS(I,model,OPTION1,OPTION2,...) performs LPS feature extraction
% with specified options. Option arguments can have the following values:
%
%   'winsize'    The size of window based on which histogram is calculated.
%                Default value is 16.
%
%   'step'       The distance between two neighboring landmarks. If 'landmarks'
%                option is specified, the specification of 'step' will be
%                ignored. Default value is 8.
%
%   'landmarks'  A matrix specifying the location of landmarks with each
%                row containing a coordinate pair (x,y) where x is the row
%                location and y is the column location. By default, the
%                landmark is calculated by 'step' option.
%
% See also PRMI_LPS_TRAIN
%
% Contact: www.dihong.me

if length(varargin)<2
    error('prmi_lps: At least two arguments required.');
else
    I = varargin{1};
    model = varargin{2};
end

validateattributes(I,{'uint8'},{'nonsparse','finite','2d'},mfilename,'I',1);

allStrings = {'winsize', 'step', 'landmarks'};

%defaults
winsize = 16;
step = 8;
landmarks = [];

%parse inputs.
for k = 3:2:length(varargin)
  if ischar(varargin{k})
    string = lower(validatestring(varargin{k}, allStrings, mfilename, 'OPTION',k));
    switch string
        case {'winsize'}
            validateattributes(varargin{k+1},{'double'}, {'positive'},mfilename,'winsize',1+k);
            winsize = varargin{k+1};
        case {'step'}
            validateattributes(varargin{k+1},{'double'}, {'positive'},mfilename,'step',1+k);
            step = varargin{k+1};
        case {'landmarks'}
            validateattributes(varargin{k+1},{'double'}, {'positive','nrows',2},mfilename,'landmarks',1+k);
            landmarks = varargin{k+1};
    end
  else
      error('prmi_lps_encode: incorrect option format.');
  end
end

%run the LPS.
if nargout==1
    F = LPS_test(I,winsize,step,landmarks,model);
elseif nargout==3
    [F,P] = LPS_test(I,winsize,step,landmarks,model);
elseif nargout==4
    [F,P,Icode] = LPS_test(I,winsize,step,landmarks,model);
end
Ft = reshape(cat(2,F{:}),[],1);
end