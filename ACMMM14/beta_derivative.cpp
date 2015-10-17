#include "mex.h"
#include"stdio.h"
#include"math.h"
#include"stdlib.h"
#include"memory.h"
using namespace std;

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {
    if(nrhs!=2)
        mexErrMsgTxt("3 inputs required.");
    else if(nlhs>1)
        mexErrMsgTxt("Too many output arguments");
    //extract input arguments.
    double* A = mxGetPr(prhs[0]);//A
    double* X = mxGetPr(prhs[1]);//X: NxM 
    int N = mxGetN(prhs[1]); //#samples
    int M = mxGetM(prhs[1]); //dim
    
    plhs[0]=mxCreateDoubleMatrix(M,1,mxREAL);
    double* out = mxGetPr(plhs[0]);//output pointer
    memset(out,0,sizeof(double)*M);

    int i,j,k;
    double t1,*t2,*t3;
    for(i = 0;i<N;i++){
        for(j = i+1;j<N;j++){
            t1 = A[i*N+j];
            t2 = X+i*M;
            t3 = X+j*M;
            for(k = 0;k<M;k++)
                out[k] += t1*pow(t2[k]-t3[k],2);
        }
    }
    for(k = 0;k<M;k++) out[k] *= 2.0;
}