function [Ft,F,P,Icode] = prmi_mlbp(varargin)
% PRMI_MLBP Multi-scale Local Binary Patterns (MLBP) feature extraction.
% [Ft,F,P,Icode] = PRMI_MLBP(I) performs MLBP feature extraction on 2-D image I.
% Feature Ft is Dx1 dimensional feature. F is a cell array with each element 
% contaning a feature matrix where each column is the local feature extracted 
% at the corresponding column of P which specifies the landmark location of 
% that local MLBP feature. Icode is the cell array containing encoded images.
%
% [Ft,F,P] = PRMI_MLBP(I,OPTION1,OPTION2,...) performs MLBP feature extraction
% with specified options. Option arguments can have the following values:
%
%   'winsize'  The size of window based on which histogram is calculated.
%              Default value is 16.
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
%   'encoding' The encoding scheme for MLBP. 
%              'u2'   for uniform LBP (default)
%              'ri'   for rotation-invariant LBP
%              'riu2' for uniform rotation-invariant LBP.
%              'none' for raw LBP features.
%
%   'scales'   An array specifies the scales of MLBP. Default is [1,3,5,7].
%
% See also PRMI_DSIFT, PRMI_HOG, PRMI_BIF
%
% Contact: www.dihong.me

if length(varargin)<1
    error('prmi_mlbp: At least one argument required.');
else
    I = varargin{1};
end

validateattributes(I,{'uint8'},{'nonsparse','finite','2d'},mfilename,'I',1);

allStrings = {'winsize', 'step', 'landmarks', 'encoding','scales'};

%defaults
winsize = 16;
step = 8;
landmarks = [];
encoding = 'u2';
scales = [1 3 5 7];

%parse inputs.
for k = 2:2:length(varargin)
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
        case {'encoding'}
            encoding = validatestring(varargin{k+1}, {'u2','ri','riu2','none'}, mfilename, 'OPTION',k);
        case {'scales'}
            validateattributes(varargin{k+1},{'double'}, {'positive'},mfilename,'numbins',1+k);
            scales = varargin{k+1};
    end
  else
      error('prmi_mlbp: incorrect option format.');
  end
end
if strcmpi(encoding,'none')
    table = 1:256;
    nb_codes = 256;
else
    m = getmapping(8,encoding);
    table = m.table;
    nb_codes = m.num;
end
%run the MLBP.
if nargout == 4
	[F,P,Icode] = MLBP(I,winsize,step,int32(landmarks'),int32(table),nb_codes,int32(scales));
    for i = 1:length(Icode)
        Icode{i} = reshape(table(double(Icode{i})+1),size(Icode{i}));
    end
elseif nargout == 3
	[F,P] = MLBP(I,winsize,step,int32(landmarks'),int32(table),nb_codes,int32(scales));
else
	F = MLBP(I,winsize,step,int32(landmarks'),int32(table),nb_codes,int32(scales));
end
Ft = reshape(cat(2,F{:}),[],1);
if nargout==1
    clear F;
end
end

%%
function mapping = getmapping(samples,mappingtype)
table = 0:2^samples-1;
newMax  = 0; %number of patterns in the resulting LBP code
index   = 1;

if strcmp(mappingtype,'u2') %Uniform 2
  newMax = samples*(samples-1) + 3;%total number of all patterns: 8*(8-1)+"00000000"+"11111111"+ over-transition.
  
  for i = 0:2^samples-1
    j = bitset(bitshift(i,1,'uint8'),1,bitget(i,samples)); %rotate left
    numt = sum(bitget(bitxor(i,j),1:samples)); %number of 1->0 and
                                               %0->1 transitions
                                               %in binary string 
                                               %x is equal to the
                                               %number of 1-bits in
                                               %XOR(x,Rotate left(x)) 
    if numt <= 2
      table(i+1) = index;
      index = index + 1;
    else
      table(i+1) = newMax;
    end
  end
  
end

if strcmp(mappingtype,'ri') %Rotation invariant
  tmpMap = zeros(2^samples,1) - 1;
  for i = 0:2^samples-1
    rm = i;
    r  = i;
    for j = 1:samples-1
      r = bitset(bitshift(i,1,'uint8'),1,bitget(r,samples)); %rotate
                                                             %left
      if r < rm
        rm = r;
      end
    end
    if tmpMap(rm+1) < 0
      tmpMap(rm+1) = newMax;
      newMax = newMax + 1;
    end
    table(i+1) = tmpMap(rm+1);
  end
end

if strcmp(mappingtype,'riu2') %Uniform & Rotation invariant
  newMax = samples + 2;
  for i = 0:2^samples - 1
    j = bitset(bitshift(i,1,'uint8'),1,bitget(i,samples)); %rotate left
    numt = sum(bitget(bitxor(i,j),1:samples));
    if numt <= 2
      table(i+1) = sum(bitget(i,1:samples));
    else
      table(i+1) = samples+1;
    end
  end
end

mapping.table=table;
mapping.samples=samples;
mapping.num=newMax;
end

