//lbp2.cpp: this is a LBP encoding function.
//input: Image, scales, szScale, szPatch, facSift, [width height], lenCode.
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

int* GetImageCode(double*I,int height,int width,int SamplingRadii);

//ofstream debug("debug.txt");

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
/* Check for proper number of arguments. */
if(nrhs!=5) 
	mexErrMsgTxt("5 inputs required.");
else if(nlhs>2)
	mexErrMsgTxt("Too many output arguments");
//extract input arguments.
double* I=mxGetPr(prhs[0]);//input encoded image.
int szPatch=(int)*mxGetPr(prhs[1]);//patch size
double facSift=*mxGetPr(prhs[2]);
double* Szd=mxGetPr(prhs[3]);//size of image.
int lenCode=(int)*mxGetPr(prhs[4]);
int height=(int)Szd[0];
int width=(int)Szd[1];
int szPatchHalf=(int)(szPatch/2);
int szSift=(int)(szPatch*facSift);

int numPatchX=1+(height-(height-szPatch)%szSift-szPatch)/szSift;
int numPatchY=1+(width-(width-szPatch)%szSift-szPatch)/szSift;

//int X0=(int)((height-szSift*(numPatchX-2))/2);
//int Y0=(int)((width-szSift*(numPatchY-2))/2);
int X0=szPatchHalf+(int)(((height-szPatch)%szSift)/2);
int Y0=szPatchHalf+(int)(((width-szPatch)%szSift)/2);
//create output space.
plhs[0]=mxCreateDoubleMatrix(numPatchX*numPatchY*lenCode,1,mxREAL);
//plhs[1]=mxCreateDoubleMatrix(250,200,mxREAL);
double* y=mxGetPr(plhs[0]);//output pointer
//double* yy=mxGetPr(plhs[1]);
memset(y,0,sizeof(double)*numPatchX*numPatchY*lenCode);//clear zero.
//-----------------------------------------------------------------------------------------------------------------
int cnt=0,r,q,i,j,m;
double max;
	for(i=0;i<numPatchX;i++){
		for(j=0;j<numPatchY;j++){
			for(r=i*szSift+X0-szPatchHalf;r<i*szSift+X0+szPatchHalf;r++){
				for(q=j*szSift+Y0-szPatchHalf;q<j*szSift+Y0+szPatchHalf;q++){
						y[cnt+(int)I[q*height+r]]++;
				}
            }
            max=0;
            for(r=0;r<lenCode;r++)
                if(y[cnt+r]>max)
                    max=y[cnt+r];
            for(r=0;r<lenCode;r++)
                y[cnt+r]=y[cnt+r]/max;
            for(r=0;r<lenCode;r++)
                y[cnt+r]=log(1+sqrt(y[cnt+r])+1e-5);
            
			cnt=cnt+lenCode;
		}
	}
}


