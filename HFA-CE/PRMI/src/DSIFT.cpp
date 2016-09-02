
//g++ DSIFT.cpp -o DSIFT -lvl

extern "C" {
  #include <prmi/vl/generic.h>
  #include <prmi/vl/dsift.h>
}
#include "mex.h"
#include"math.h"
#include"stdlib.h"
#include"memory.h"

using namespace std;

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {
    //extract input arguments.
    unsigned char* Image = (unsigned char*)mxGetPr(prhs[0]);//input image.
    int winsize = *mxGetPr(prhs[1]);//winsize
    int step = *mxGetPr(prhs[2]);//step
    int numBin = *mxGetPr(prhs[3]); //nbins
    
    int height = mxGetM(prhs[0]);
    int width = mxGetN(prhs[0]);
    
    //features configurations.
    VlDsiftFilter * vf = vl_dsift_new (width,height);
    vl_dsift_set_steps(vf,step,step);
    VlDsiftDescriptorGeometry geom;
    geom.numBinT = numBin;
    geom.numBinX = 4;
    geom.numBinY = 4;
    geom.binSizeX = winsize;
    geom.binSizeY = winsize;
    vl_dsift_set_geometry(vf,&geom) ;
    vl_dsift_set_flat_window(vf,true);
    //extract features.
    float* im = (float*)mxMalloc(sizeof(float)*height*width);
    for(int i = 0;i<height;i++)
        for(int j=0;j<width;j++)
            im[i*width+j] = Image[j*height+i];
    vl_dsift_process (vf,im);
    const float* feat = vl_dsift_get_descriptors(vf);
    int length = vl_dsift_get_keypoint_num(vf)*vl_dsift_get_descriptor_size(vf);
    
    plhs[0]=mxCreateNumericMatrix(length,1, mxSINGLE_CLASS,mxREAL);
    memcpy(mxGetPr(plhs[0]),feat,sizeof(float)*length);
    
    vl_dsift_delete(vf);
    mxFree(im);
}
