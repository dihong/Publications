//This function extract the pixel features for each input image.
//the output is a matrix organized as: [patch*8*d]x[Np*Np]
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

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
/* Check for proper number of arguments. */
if(nrhs!=5) 
	mexErrMsgTxt("5 inputs required.");
else if(nlhs>2)
	mexErrMsgTxt("Too many output arguments");
//extract input arguments.
double* I=mxGetPr(prhs[0]);//input image.
int radii=(int)*mxGetPr(prhs[1]); //sampling radii
double* Szd=mxGetPr(prhs[2]);//size of image.
double* Rx = mxGetPr(prhs[3]); //sampling pattern X.
double* Ry = mxGetPr(prhs[4]); //sampling pattern Y.

int H=(int)Szd[0];
int W=(int)Szd[1];
int d = 8*radii; //dimension of pixel vectors.

//create output space.
plhs[0]=mxCreateDoubleMatrix(d,H*W,mxREAL);
double* y=mxGetPr(plhs[0]);//output pointer
memset(y,0,sizeof(double)*d*H*W);//clear zero.

int cp, sx, sy, cnt=0;
for(int i = 0;i<W;i++){
	for(int j = 0;j<H;j++){
		cp = I[i*H+j];
		for(int k=0;k<d;k++){
			//for each surrounding component.
			sx = i + Rx[k];
			sy = j - Ry[k];
			if(sx>=0 && sx<W && sy>=0 && sy<H)
				y[cnt*d+k] = I[sx*H+sy]-cp;
			else
				y[cnt*d+k] = 0;
		}
		cnt++;
	}
}
}