function feat = lbpExtract(szPatch,sift_fac,scales,faces)
%%
table=getmapping(8,'u2');
table=double(table.table);
lenCode=59;
szImage = size(faces{1});
parfor indImg =1 : length(faces)
    if numel(faces{indImg})>0
        feat{indImg} = PMLBP(faces{indImg}, scales, szPatch, sift_fac, szImage, table, lenCode);
    end
end
feat = cat(2,feat{:});
end

function f = PMLBP(face, scales, szPatch, sift_fac, szImage, table, lenCode)
f = MLBP(double(face),scales,length(scales),szPatch,sift_fac,szImage,table,lenCode);
end