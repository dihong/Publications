//HOG.cpp: this is a HOG encoding function.
//input: Image, szPatch, facSift, [width height],blockFilter
//output: 1*N array, N is the number of sampled points.
//###################################################

#include "mex.h"
#include"stdio.h"
#include"math.h"
#include"stdlib.h"
#include"memory.h"
using namespace std;

#define pi 3.1416


// ofstream debug("debug.txt");

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {
    if(nrhs!=3)
        mexErrMsgTxt("3 inputs required.");
    else if(nlhs>1)
        mexErrMsgTxt("Too many output arguments");
    //extract input arguments.
    double* Image = mxGetPr(prhs[0]);//input image.
    int szPatch=(int)*mxGetPr(prhs[1]);//block size
    int szSift = *mxGetPr(prhs[2]);//step
    int height=mxGetM(prhs[0]);
    int width=mxGetN(prhs[0]);
    int szPatchHalf=(int)(szPatch/2);
    int numPatchX=1+(height-(height-szPatch)%szSift-szPatch)/szSift;
    int numPatchY=1+(width-(width-szPatch)%szSift-szPatch)/szSift;
    int numBin=12;
    if(szPatch%8!=0)
        mexErrMsgTxt("size of patch must be 8-multiple.");
    plhs[0]=mxCreateDoubleMatrix(numPatchX*numPatchY*16*numBin, 1, mxREAL);
    double* y=mxGetPr(plhs[0]);//output pointer
    if(szPatch>height || szPatch>width)
        mexErrMsgTxt("szPatch>height || szPatch>width");
//-----------------------------------------------------------------------------------------------------------------
    int i, j, k;
//=========convert input double image to int image to improve efficiency==========
    int* I=new int [height*width];
    for(i=0;i<height*width;i++)
        I[i]=(int)Image[i];
//=========compute gradient of Image===================
    float dx, dy, theta, mag, dx2, dy2;
    float* IgradMag=new float[height*width];
    int* IgradOrient=new int[height*width]; //0->numBin.
    for(i=0;i<height;i++){// compute the gradient of image.
        for (j=0;j<width;j++){
            if(i==height-1)
                dx=I[j*height+height-1]-I[j*height+height-2];
            else if(i==0)
                dx=I[j*height+1]-I[j*height];
            else{
                if (j>0 && j< width-1)
                    dx=((I[(j+1)*height+i+1]+I[(j)*height+i+1]+I[(j-1)*height+i+1])
                    -(I[(j+1)*height+i-1]+I[(j)*height+i-1]+I[(j-1)*height+i-1]))/3.0;
                else
                    dx=I[j*height+i+1]-I[j*height+i-1];
            }
            if(j==width-1)
                dy=I[(width-1)*height+i]-I[(width-2)*height+i];
            else if (j==0)
                dy=I[height+i]-I[i];
            else{
                if (i>0 && i< height-1)
                    dy=((I[(j+1)*height+i+1]+I[(j+1)*height+i]+I[(j+1)*height+i-1])
                    -(I[(j-1)*height+i+1]+I[(j-1)*height+i]+I[(j-1)*height+i-1]))/3.0;
                else
                    dy=I[(j+1)*height+i]-I[(j-1)*height+i];
            }
            mag=sqrt(pow(dx, 2)+pow(dy, 2));
            IgradMag[j*height+i]=mag;
            if(IgradMag[j*height+i]==0){
                IgradOrient[j*height+i]=0;
                continue;
            }
            if (dy>=0)
                theta=acos(dx/mag);
            else if(dy<0)
                theta=2*pi-acos(dx/mag);
            IgradOrient[j*height+i]=(int)(0.5+theta*numBin/2/pi)%numBin;//0->numBin-1.
        }
    }
//============compute weighted gradient orientation histogram=================
    float* orientHist=new float [numPatchX*numPatchY*16*numBin];
    memset(orientHist, 0, sizeof(float)*numPatchX*numPatchY*16*numBin); //initialization.
    int Xoff=(int)(((height-szPatch)%szSift)/2);
    int Yoff=(int)(((width-szPatch)%szSift)/2);
    int cellStep=(int)(szPatch/4);
    int cellStepHalf=(int)(cellStep/2);
    int pixelX, pixelY, seedX, seedY;
    float cellX, cellY, cellMagEng, blockMagEng, weightedIgradMag;
    int cntBlock=0; //current block id.
    int cntCell=0;//current cell id within current block.
    for(i=0;i<numPatchX;i++){
        for(j=0;j<numPatchY;j++){
            //block level.
            seedX=Xoff+i*szSift+szPatchHalf;
            seedY=Yoff+j*szSift+szPatchHalf;
            blockMagEng=0;cntCell=0;
            for(float cx=-1.5;cx<=1.5;cx+=1){
                for(float cy=-1.5;cy<=1.5;cy+=1){
                    //cell level:4*4=16 cells in each block.
                    cellX=cx*cellStep+seedX;
                    cellY=cy*cellStep+seedY;
                    cellMagEng=0;
                    for(float px=-cellStepHalf+0.5;px<=cellStepHalf-0.5;px+=1){
                        //pixel level: cellStep*cellStep pixels in each cell. statistic accumulated energy distribution within cells.
                        for(float py=-cellStepHalf+0.5;py<=cellStepHalf-0.5;py+=1){
                            pixelX=int(px+cellX);
                            pixelY=int(py+cellY);
                            orientHist[cntBlock*numBin*16+cntCell*numBin+IgradOrient[pixelY*height+pixelX]]+=IgradMag[pixelY*height+pixelX];
                            cellMagEng+=IgradMag[pixelY*height+pixelX];
                        }
                    }
                    blockMagEng=blockMagEng+cellMagEng;
                    cntCell++;
                }
            }
            //====normalization====
            for(k=0; k<16*numBin; k++)
                if(blockMagEng)
                    orientHist[cntBlock*numBin*16+k]/=blockMagEng;//within-block energy normalization.
            
            cntBlock++;
        }
    }
    
    
//=====return result=====
    for(k=0;k<numPatchX*numPatchY*16*numBin;k++)
        y[k]=log(1+sqrt((orientHist[k])));
    
//=====free memory======
    delete I, IgradMag, IgradOrient, orientHist;
}
