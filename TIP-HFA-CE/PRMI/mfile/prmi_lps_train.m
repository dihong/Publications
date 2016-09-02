function model = prmi_lps_train(varargin)
% PRMI_LPS_TRAIN Learning Local Pattern Selection (LPS) feature extraction model.
% model = PRMI_LPS(Images1,Images2) performs LPS learning based on Images1 and Images2.
% Images1 is a 1xN cell array containing the first training face images from N
% subjects, and Images2 is a 1xN cell array containing the second training
% faces corresponding to the subjects in the Images1.
%
% model = PRMI_LPS_TRAIN(Images1,Images2,OPTION1,OPTION2,...) performs LPS
% learning with specified options. Option arguments can have the following values:
%
%   'ncode'      The number of codes for LPS. Default ncode = 64.
%
%   'alpha'      The (0,1) tradeoff factor. Default alpha = 0.5.
%
%   'scales'     An array specifies the scales of LPS. Default is [1,3,5,7].
%
%	'align'		 If align faces before training, set align = 1, otherwise
%				 set align = 0. Default value is 0.
%
%   '-v'         Verbose mode. Print training status. Default not printing.
%
% See also PRMI_LPS
%
% Contact: www.dihong.me

if length(varargin)<2
    error('prmi_lps: At least two arguments required.');
else
    Images1 = varargin{1};
    Images2 = varargin{2};
end

validateattributes(Images1,{'cell'},{},mfilename,'Images1',1);
for i = 1:length(Images1)
    validateattributes(Images1{i},{'uint8'},{'nonsparse','finite','2d'},mfilename,['Images1{' num2str(i) '}'],1);
end

validateattributes(Images2,{'cell'},{},mfilename,'Images2',2);
for i = 1:length(Images2)
    validateattributes(Images2{i},{'uint8'},{'nonsparse','finite','2d'},mfilename,['Images2{' num2str(i) '}'],2);
end

if length(Images1) ~= length(Images2)
    error('The size of Images1 and Images2 must be the same.');
end

allStrings = {'ncode','alpha','scales','align','-v'};

%defaults
ncode = 64;
alpha = 0.5;
scales = [1 3 5 7];
align = 0;
verbose = 0;

%parse inputs.
k = 3;
while k <= length(varargin)
  if ischar(varargin{k})
    string = lower(validatestring(varargin{k}, allStrings, mfilename, 'OPTION',k));
    switch string
        case {'ncode'}
            validateattributes(varargin{k+1},{'double'}, {'positive','integer'},mfilename,'ncode',1+k);
            ncode = varargin{k+1};
        case {'alpha'}
            validateattributes(varargin{k+1},{'double'}, {'positive','>',0,'<',1},mfilename,'alpha',1+k);
            alpha = varargin{k+1};
        case {'scales'}
            validateattributes(varargin{k+1},{'double'}, {'positive'},mfilename,'numbins',1+k);
            scales = varargin{k+1};
        case {'align'}
            validateattributes(varargin{k+1},{'double'}, {'integer','>=',0,'<=',1},mfilename,'align',1+k);
            align = varargin{k+1};
        case {'-v'}
            verbose = 1;
            k = k - 1;
        otherwise
                error(['Unknown parameter: ' string]);
    end
    k = k + 2;
  else
      error('prmi_lps: incorrect option format.');
  end
end
%train the model.
if verbose == 1
    model = LPS_train(Images1,Images2,[ncode 0],scales,alpha,align);
else
    model = LPS_train(Images1,Images2,ncode,scales,alpha,align);
end
end
