%% This function groups the TRAINING faces according to their ages.
% ret: a KxN matrix with each column contains a 8x1 vector of elements 0,
% 1, 2 or 3. 0 means no face at this age group, 1 means train1 has one face
% at this group and similarly for 2. While 3 means both train1 and train2
% have faces at this group.
% matrix.

function ret = grouping_on_ages(path1,path2,K,N)
% path1 = 'train1';
% path2 = 'train2';
% K = 2;

fls1 = dir([path1 '\*.bmp']);
fls2 = dir([path2 '\*.bmp']);
if nargin == 3
    N = length(fls1);
end
ret = zeros(K,N);
H = zeros(1,100); %histogram of the age distribution.
for i = 1:N
    name = fls1(i).name; % name of the subject.
    ind = strfind(name,'.bmp')-2;
    age = str2double(name(ind:ind+1)); %retrieve the age according to the file name.
    H(age) = H(age) + 1;
    name = fls2(i).name;
    ind = strfind(name,'.bmp')-2;
    age = str2double(name(ind:ind+1));
    H(age) = H(age) + 1;
end

%compute the pivot which splits the training samples into K groups as even
%as possible.
C = zeros(1,100);
for i = 1:100
    C(i) = sum(H(1:i));
end

S = max(C)/K;
for k = 1:K-1
    [val pivot(k)] = min(abs(C-S));
    C = C-C(pivot(k));
    C(C<=0) = 0; 
end
pivot(K) = 100;
%grouping according to 'pivot'.
for i = 1:N
    name = fls1(i).name;
    ind = strfind(name,'.bmp')-2;
    age = str2double(name(ind:ind+1));
    ind = find(age<=pivot);
    gid1 = ind(1); %falls into which group.
    name = fls2(i).name;
    ind = strfind(name,'.bmp')-2;
    age = str2double(name(ind:ind+1));
    ind = find(age<=pivot);
    gid2 = ind(1); %falls into which group.
    if gid1==gid2
        ret(gid1,i) = 3;
    else
        ret(gid1,i) = 1;
        ret(gid2,i) = 2;
    end
end
end