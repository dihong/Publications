function Ft = prmi_multi_features(varargin)
% PRMI_MULTI_FEATURES Multi-features funsion.
% Ft = PRMI_MULTI_FEATURES(I) performs multi-feature extraction on 2-D image I, 
% which fuses features of DSIFT and MLBP with their default parameters.
% Feature Ft is Dx1 dimensional concatenated feature.
% 
% Ft = PRMI_MULTI_FEATURES(I,OPTION1,OPTION2,...) performs multi-feature extraction
% with specified options. Option arguments can have the following values:
%
%   'dsift'  Incorporate DSIFT features as output. It is a cell array
%            specifies the parameters of dsift parameters. If the parameters
%            are not specified, then it will use the default settings of
%            DSIFT descriptor.
%
%   'hog'    Incorporate HOG features as output. It is a cell array
%            specifies the parameters of dsift parameters. If the parameters
%            are not specified, then it will use the default settings of
%            HOG descriptor.
%
%   'mlbp'   Incorporate MLBP features as output. It is a cell array
%            specifies the parameters of dsift parameters. If the parameters
%            are not specified, then it will use the default settings of
%            MLBP descriptor.
%
%   'lps'    Incorporate LPS features as output. It is a cell array
%            specifies the parameters of dsift parameters. If the parameters
%            are not specified, then it will use the default settings of
%            LPS descriptor.
%
%   'cfda'   Incorporate CFDA features as output. It is a cell array
%            specifies the parameters of dsift parameters. If the parameters
%            are not specified, then it will use the default settings of
%            CFDA descriptor.
%
%   'cefd'   Incorporate CEFD features as output. It is a cell array
%            specifies the parameters of dsift parameters. If the parameters
%            are not specified, then it will use the default settings of
%            CEFD descriptor.
%
% Examples:
%
%    Example 1: extract default multi-features (dsift+mlbp).
%            I = imresize(rgb2gray(imread('peppers.png')),[100 NaN]);
%            Ft = prmi_multi_features(I);
%
%    Example 2: extract multi-features of 'hog+mlbp' with default settings.
%            I = imresize(rgb2gray(imread('peppers.png')),[100 NaN]);
%            Ft = prmi_multi_features(I,'hog','mlbp');
%
%    Example 3: extract multi-features of 'hog+mlbp' with customized settings.
%            I = imresize(rgb2gray(imread('peppers.png')),[100 NaN]);
%            hog_settings = {'winsize',20,'nbins',8};
%            Ft = prmi_multi_features(I,'hog',hog_settings,'mlbp');
%
% See also PRMI_MLBP, PRMI_HOG, PRMI_DSIFT
%
% Contact: www.dihong.me

if length(varargin)<1
    error('prmi_multi_features: At least one argument required.');
else
    I = varargin{1};
end

validateattributes(I,{'uint8'},{'nonsparse','finite','nonnan','2d'},mfilename,'I',1);

allStrings = {'dsift', 'hog', 'hog2', 'mlbp','lps'};

if length(varargin) > 1
    %parse inputs.
    k = 2;
    descriptors = {};
    settings = {};
    while k <= length(varargin)
        if ischar(varargin{k})
            string = lower(validatestring(varargin{k}, allStrings, mfilename, 'OPTION',k));
            switch string
                case {'dsift'}
                    descriptors = [descriptors {@prmi_dsift}];
                    if k== length(varargin) || ischar(varargin{k+1})
                        settings{end+1} = {};
                    else
                        validateattributes(varargin{k+1},{'cell'}, {},mfilename,'dsift',1+k);
                        settings{end+1} = varargin{k+1};
                        k = k + 1;
                    end
                case {'hog'}
                    descriptors = [descriptors {@prmi_hog}];
                    if k== length(varargin) || ischar(varargin{k+1})
                        settings{end+1} = {};
                    else
                        validateattributes(varargin{k+1},{'cell'}, {},mfilename,'hog',1+k);
                        settings{end+1} = varargin{k+1};
                        k = k + 1;
                    end
                case {'hog2'}
                    descriptors = [descriptors {@prmi_hog2}];
                    if k== length(varargin) || ischar(varargin{k+1})
                        settings{end+1} = {};
                    else
                        validateattributes(varargin{k+1},{'cell'}, {},mfilename,'hog2',1+k);
                        settings{end+1} = varargin{k+1};
                        k = k + 1;
                    end
                case {'mlbp'}
                    descriptors = [descriptors {@prmi_mlbp}];
                    if k== length(varargin) || ischar(varargin{k+1})
                        settings{end+1} = {};
                    else
                        validateattributes(varargin{k+1},{'cell'}, {},mfilename,'mlbp',1+k);
                        settings{end+1} = varargin{k+1};
                        k = k + 1;
                    end
                case {'lps'}
                    descriptors = [descriptors {@prmi_lps}];
                    if k== length(varargin) || ischar(varargin{k+1})
                        settings{end+1} = {};
                    else
                        validateattributes(varargin{k+1},{'cell'}, {},mfilename,'lps',1+k);
                        settings{end+1} = varargin{k+1};
                        k = k + 1;
                    end
            end
        else
            error('prmi_multi_features: incorrect option format.');
        end
        k = k + 1;
    end    
else
    %defaults
    descriptors = {@prmi_dsift,@prmi_mlbp};
    settings = {{},{}};
end


Ft = [];
for i = 1:length(descriptors)
    Ft = [Ft; feval(descriptors{i},I,settings{i}{:})];
end
end