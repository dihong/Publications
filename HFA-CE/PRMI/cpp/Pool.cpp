//HOG.cpp: this is a HOG encoding function.
//input: Image, szPatch, facSift, [width height],blockFilter
//output: 1*N array, N is the number of sampled points.
//###################################################

#include "mex.h"
#include"stdio.h"
#include"math.h"
#include"stdlib.h"
#include"memory.h"
#include<fstream>
#include<ctime>
using namespace std;

#define pi 3.1416


// ofstream debug("debug.txt");

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {
    if(nrhs!=3)
        mexErrMsgTxt("3 inputs required.");
    else if(nlhs>1)
        mexErrMsgTxt("Too many output arguments");
    //extract input arguments.
    double* Image=mxGetPr(prhs[0]);//input image.
    int szPatch=(int)*mxGetPr(prhs[1]);//block size
    double* Szd=mxGetPr(prhs[2]);//size of image.
    int height=(int)Szd[0];
    int width=(int)Szd[1];
    
    float facSift = 0.5;
    int szSift=(int)(szPatch*facSift);
    int szPatchHalf=(int)(szPatch/2);
    int numPatchX=1+(height-(height-szPatch)%szSift-szPatch)/szSift;
    int numPatchY=1+(width-(width-szPatch)%szSift-szPatch)/szSift;
    

    plhs[0]=mxCreateDoubleMatrix(numPatchX*numPatchY, 1, mxREAL);
    double* y=mxGetPr(plhs[0]);//output pointer
    if(szPatch>height || szPatch>width)
        mexErrMsgTxt("szPatch>height || szPatch>width");
//-----------------------------------------------------------------------------------------------------------------
    int i, j, p, q, seedX, seedY;
    int Xoff=(int)(((height-szPatch)%szSift)/2);
    int Yoff=(int)(((width-szPatch)%szSift)/2);
    double avg,var;
    for(i=0;i<numPatchX;i++){
        for(j=0;j<numPatchY;j++){
            seedX=Xoff+i*szSift+szPatchHalf;
            seedY=Yoff+j*szSift+szPatchHalf;
            avg = 0;
            int cnt = 0;
            for(p=seedX-szPatchHalf;p<seedX+szPatchHalf;p++){
                for(q=seedY-szPatchHalf;q<seedY+szPatchHalf;q++){
                    avg += Image[q*height+p];
                    cnt++;
                }
            }
            avg /= cnt;
            var = 0;
            for(p=seedX-szPatchHalf;p<seedX+szPatchHalf;p++){
                for(q=seedY-szPatchHalf;q<seedY+szPatchHalf;q++){
                    var += (Image[q*height+p]-avg)*(Image[q*height+p]-avg);
                }
            }
            y[j*numPatchX+i] = sqrt(var/cnt);
        }
    }
}