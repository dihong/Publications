%
%% This is the main entry of the program.
% #Inputs:
%       path1 - the path of the 2800 training/testing persons of photography.
%       path2 - the path of the 2800 training/testing persons of near-infrared images.

% #Output: rank_1 is the rank-1 recognition rate.

% #Comments: 'CoreNum' requires to be no more than the maximum MATLAB workers
% available in your local system.

% Auothr: Dihong Gong

% ============================Copyright=================================
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 1. Redistributions of source code must retain the above copyright
%    notice, this list of conditions and the following disclaimer.
% 2. Redistributions in binary form must reproduce the above copyright
%    notice, this list of conditions and the following disclaimer in the
%    documentation and/or other materials provided with the
%    distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
% "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
% A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
% HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
% SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
% LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
% DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
% THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function rank_1 = main(path1,path2)
CoreNum=12;
MATLABPOOL=1;
if ~MATLABPOOL && (matlabpool('size')>0)
    matlabpool close;
end
if MATLABPOOL
    if(matlabpool('size')~=CoreNum && matlabpool('size')>0)
        matlabpool close;
        matlabpool('open',CoreNum) %open local.
    elseif(matlabpool('size')<1)
        matlabpool('open',CoreNum) %open local.
    end
end
%%
winsize = {[150 120]};
radii = {3,5,7,9};
szpatch = 30;
sift = 0.3;
for i = 1:length(winsize)
    for j = 1:length(radii)
        fprintf('radii = %d\n',radii{j});
        cvt2code_linear(path1,path2,winsize{i},radii{j});
    end
end
rank_1 = test(winsize,radii,szpatch,sift);
end