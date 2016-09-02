#include "mex.h"
#include"math.h"
#include"stdlib.h"
#include"memory.h"
#define pi 3.14159


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

float* local_feature_hog(unsigned char* Image, const int* landmarks, const int& N, const int winsize, const int& height, const int& width, const int& numBin){
    const int* X = landmarks;
    const int* Y = landmarks + N;
    //= == == == == compute gradient of Image == == == == == == == == == =
    float dx, dy, theta, mag;
    float* IgradMag = (float*) mxMalloc(sizeof(float)*height * width);
    int* IgradOrient = (int*) mxMalloc(sizeof(int)*height * width);
    memset(IgradOrient,0,sizeof(int)*width*height);
    int i,j,ind;
    int* I = (int*)mxMalloc(sizeof(int)*height*width);
    for(i=0;i<height*width;i++)
        I[i]=(int)Image[i];
    for (i = 0; i < height; i++){// compute the gradient of image.
        for (j = 0; j < width; j++) {
            ind = i*width+j;
            if (i == height - 1)
                dy = I[(height-1)+j*height] - I[(height-2)+j*height];
            else if (i == 0)
                dy = I[1+j*height] - I[j*height];
            else
                dy = (I[(i+1)+j*height] - I[(i-1)+j*height])/2.0;
            
            if (j == width - 1)
                dx = I[(width-1)*height+i] - I[(width-2)*height+i];
            else if (j == 0)
                dx = I[height+i] - I[i];
            else
                dx = (I[i+(j+1)*height] - I[i+(j-1)*height])/2.0;
            mag = sqrt(dx*dx+dy*dy);
            IgradMag[ind] = mag;
            if (mag == 0){
                IgradOrient[ind] = numBin/2;
                continue;
            }
            if (dy >= 0)
                theta = acos(dx / mag);
            else if (dy < 0)
                theta = 2 * pi - acos(dx / mag);
            IgradOrient[ind] = ((int)(0.5 + theta * numBin / 2 / pi)) % numBin; //0->numBin-1.
        }
    }
    //============compute gradient orientation histogram=================
    int lenfeat = 4*numBin;
	int sizeBlock = 4*winsize;
    float* feat = (float*) mxMalloc(sizeof(float)*N*lenfeat); //each row is a local feature.
    memset(feat, 0, sizeof(float)*N*lenfeat); //initialization.
    int pixelX, pixelY, idx1, idx2;
    float *ptr;
    int cntBlock = 0, cntCell; //current block id, cell id.
	//PRINT:
	//mexPrintf("width=%d,height=%d,sizeBlock=%d,numBin=%d,N=%d\n",width,height,sizeBlock,numBin,N);
    for (int n = 0; n < N; n++){
        //block level.
        cntCell = 0; int blockMagEng = 0;
		idx1 = cntBlock*lenfeat;
        for (int cellX = X[n]-0.5*winsize; cellX < X[n]+winsize; cellX += winsize) {
            for (int cellY = Y[n]-0.5*winsize; cellY < Y[n]+winsize; cellY += winsize){
				ptr = feat + idx1 + cntCell*numBin;
                for (int pixelX=cellX-0.5*winsize; pixelX<cellX+0.5*winsize; pixelX += 1) {
                    //pixel level: cellStep*cellStep pixels in each cell. statistic accumulated energy distribution within cells.
                    for (int pixelY=cellY-0.5*winsize; pixelY<cellY+0.5*winsize; pixelY += 1) {
                        if(pixelX>=0 && pixelX<width && pixelY>=0 && pixelY<height){
							idx2 = pixelY*width +pixelX;
                            ptr[IgradOrient[idx2]] += IgradMag[idx2];
                        }
                    }
                }
                cntCell++;
            }
        }
        cntBlock++;
    }
    mxFree(IgradMag);
    mxFree(IgradOrient);
	mxFree(I);
    for(int k= 0;k<N*lenfeat;k++)
        feat[k]=sqrt(feat[k]);
    return feat;
}
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {
    //extract input arguments.
    unsigned char* Image = (unsigned char*)mxGetPr(prhs[0]);//input image.
    int winsize = *mxGetPr(prhs[1]);//winsize
    int step = *mxGetPr(prhs[2]);//step
    int* L = (int*)mxGetPr(prhs[3]);//landmarks
    int numBin = *mxGetPr(prhs[4]);
    
    int height = mxGetM(prhs[0]);
    int width = mxGetN(prhs[0]);
    int N = mxGetM(prhs[3]);
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
    
    float* feat = local_feature_hog(Image,landmarks,N,winsize,height,width,numBin);
    
    plhs[0]=mxCreateNumericMatrix(4*numBin, N, mxSINGLE_CLASS,mxREAL);
    memcpy(mxGetPr(plhs[0]),feat,sizeof(float)*4*numBin*N);
    if(nrhs==2){
        plhs[1]=mxCreateNumericMatrix(2, N, mxSINGLE_CLASS, mxREAL);
        float* ptr = (float*)mxGetPr(plhs[1]);
        for(int i = 0;i<N;i++) ptr[i*2] = landmarks[i];
        for(int i = 0;i<N;i++) ptr[i*2+1] = landmarks[i+N];
    }
    mxFree(feat);
    mxFree(landmarks);
}
