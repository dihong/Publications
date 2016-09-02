%% This function sets up the PRMI toolbox.
function prmi_setup
%compile.
mex cpp/HOG.cpp -outdir ./mex;
mex cpp/HOG2.cpp -outdir ./mex;
mex cpp/MLBP.cpp -outdir ./mex;
mex cpp/DSIFT.cpp -lvl -outdir ./mex;
mex cpp/LPS_train.cpp cpp/luxand_face_norm.cpp cpp/parallel.cpp -lfsdk -lpthread -ldl -outdir ./mex;
mex cpp/LPS_test.cpp -outdir ./mex;
mex cpp/Pool.cpp -outdir ./mex;
end
