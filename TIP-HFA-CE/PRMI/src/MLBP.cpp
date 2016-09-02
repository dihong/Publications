#include "mex.h" 
#include"math.h"
#include"stdlib.h"
#include"memory.h"

using namespace std;

void get_fiducial_points_dense_sampling(int W, int H, int* X, int* Y, int N, int step){
    int xoff = (W%step)/2 + step;
    int yoff = (H%step)/2 + step;
    int cnt = 0;
    for(int i=xoff;i<=W-step;i+=step){
        for(int j=yoff;j<=H-step;j+=step){
            X[cnt] = i;
            Y[cnt] = j;
            cnt++;
        }
    }
}

void GetImageCode(unsigned char* I,int height,int width,int SamplingRadii,int*Icode){
	int C,xf,yf,pk,i,j,k,mainDirection,maxDirection,c;
	float xk,yk,dx,dy;
	unsigned char code;
	int Rx[]={0,-SamplingRadii,-SamplingRadii,-SamplingRadii,0,SamplingRadii,SamplingRadii,SamplingRadii};
	int Ry[]={SamplingRadii,SamplingRadii,0,-SamplingRadii,-SamplingRadii,-SamplingRadii,0,SamplingRadii};
     
	for(i=0;i<height;i++){//
		for (j=0;j<width;j++){
            C=(int)I[j*height+i];
			code=0;//decimal codes
            for(k=0;k<8;k++){
                if(i+Rx[k]>=0 && i+Rx[k]<height && j+Ry[k]>=0 && j+Ry[k]<width)
                    code = 2*code+((int)I[(j+Ry[k])*height+i+Rx[k]]>=C);//0->255.
                else
                    code *= 2;
            }
			Icode[i*width+j]=code;//0->255.
		}
	}
}

void local_feature_mlbp(unsigned char* I, int height, int width, const int* landmarks, int N, int szPatch, int* table, int nb_codes, int scale, float* feat, int* Icode){
    const int* X = landmarks;
    const int* Y = landmarks + N;
    int cnt;
    int szPatchHalf = (1+szPatch)/2;
    float max;
    GetImageCode(I,height,width,scale,Icode);
    memset(feat, 0, sizeof(float)*N*nb_codes);
    cnt = 0;
    for (int n = 0; n < N; n++){ //for each landmark point.
        for(int y = Y[n]-szPatchHalf;y < Y[n]+szPatchHalf; y++){
            for(int x = X[n]-szPatchHalf;x < X[n]+szPatchHalf; x++){
                if(x>=0 && x<width && y>=0 && y<height)
                    feat[cnt+table[Icode[y*width+x]]-1]++;
            }
        }
        cnt = cnt + nb_codes;
    }
    for(int i= 0;i<N*nb_codes;i++)
        feat[i] = log(1+feat[i]);
}




void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    unsigned char* Image = (unsigned char*)mxGetPr(prhs[0]);//input image.
    int winsize = *mxGetPr(prhs[1]);//winsize
    int step = *mxGetPr(prhs[2]);//step
    int* L = (int*)mxGetPr(prhs[3]);//landmarks
    int* table = (int*)mxGetPr(prhs[4]); //encoding table.
    int nb_codes = *mxGetPr(prhs[5]); //number of codes.
    int* scales = (int*)mxGetPr(prhs[6]);//scales.
    
    int height = mxGetM(prhs[0]);
    int width = mxGetN(prhs[0]);
    int N = mxGetM(prhs[3]);
    int nb_scales = mxGetNumberOfElements(prhs[6]);
    int* landmarks = 0;
    if (N == 0){ //need to generate landmarks.
        int numPatchX = width/step-1;
        int numPatchY = height/step-1;
        int xoff = (width%step)/2 + step;
        int yoff = (height%step)/2 + step;
        N = numPatchX*numPatchY;
        landmarks = (int*)mxMalloc(sizeof(int)*N*2);
        get_fiducial_points_dense_sampling(width, height, landmarks, landmarks+N, N, step);
    }else{
    	landmarks = (int*)mxMalloc(sizeof(int)*N*2);
		for(int i = 0;i<N;i++) landmarks[i] = L[i*2];
		for(int i = 0;i<N;i++) landmarks[i+N] = L[i*2+1];
    }
    float* feat = (float*)mxMalloc(sizeof(float)*N*nb_codes); //each row is a local feature.
    int* Icode=(int*)mxMalloc(sizeof(int)*height*width);
    mxArray *features = mxCreateCellMatrix(1,nb_scales);
    mxArray *encoded = 0;
    if(nlhs>=3) encoded = mxCreateCellMatrix(1,nb_scales);
    for(int i = 0;i<nb_scales;i++){
    	local_feature_mlbp(Image, height, width, landmarks, N, winsize, table, nb_codes, scales[i], feat, Icode);
        plhs[0] = mxCreateNumericMatrix(nb_codes, N, mxSINGLE_CLASS,mxREAL);
        memcpy(mxGetPr(plhs[0]),feat,sizeof(float)*nb_codes*N);
        mxSetCell(features, i, mxDuplicateArray(plhs[0]));
        mxDestroyArray(plhs[0]);
		if(nlhs>=3){
			plhs[2] = mxCreateNumericMatrix(height, width, mxINT32_CLASS,mxREAL);
			memcpy(mxGetPr(plhs[2]),Icode,sizeof(int)*height*width);
			mxSetCell(encoded, i, mxDuplicateArray(plhs[2]));
			mxDestroyArray(plhs[2]);
		}
    }
    if(nlhs>=3) plhs[2] = encoded;
    plhs[0] = features;
    if(nlhs>=2){
		plhs[1]=mxCreateNumericMatrix(2, N,mxSINGLE_CLASS, mxREAL);
		float* ptr = (float*)mxGetPr(plhs[1]);
		for(int i = 0;i<N;i++) ptr[i*2] = landmarks[i];
		for(int i = 0;i<N;i++) ptr[i*2+1] = landmarks[i+N];
    }
    mxFree(landmarks);
    mxFree(feat);
    mxFree(Icode);
}

