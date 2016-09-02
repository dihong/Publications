#include "mex.h" 
#include "stdlib.h"
#include "memory.h"
#include <string>
#include <vector>
#include <math.h>

#ifdef __gnu_linux__
	#include <prmi/parallel.h>
	typedef long long __int64;
#endif
using namespace std;

typedef unsigned char uchar;

#define LCHILD 0
#define RCHILD 1
#define ATB 2
#define TH 3
#define CODE 4
#define NSMPL 5


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

void GetImageCode(short* pf,int height,int width, int* Icode, double* tree, int nrows){
	int i,j,cnt=0;
	short *vec = 0;
	double *node = 0; 
	for(i=0;i<height;i++){//
		for (j=0;j<width;j++){
			vec = pf + 8*cnt;
			node = tree; //point to the root.
			while(node[CODE]==-1){ //leaf node.
				if(vec[(int)node[ATB]]<node[TH])
					node = tree + 6*(int)node[LCHILD];
				else
					node = tree + 6*(int)node[RCHILD];
			}
			Icode[cnt] = node[CODE];
			cnt++;
		}
	}
}

void GetImageCode(short* pf,int height,int width, int* Icode, int* Icode2, double* tree, int nrows){
	int i,j,cnt=0;
	short *vec = 0;
	double *node = 0; 
	for(i=0;i<height;i++){//
		for (j=0;j<width;j++){
			vec = pf + 8*cnt;
			node = tree; //point to the root.
			while(node[CODE]==-1){ //leaf node.
				if(vec[(int)node[ATB]]<node[TH])
					node = tree + 6*(int)node[LCHILD];
				else
					node = tree + 6*(int)node[RCHILD];
			}
			Icode[cnt] = node[CODE];
			Icode2[i+j*height] = node[CODE];
			cnt++;
		}
	}
}

void extract_pixel_feat(short*ret, const uchar* image, int h, int w, int radius){
	int Rx[]={0,-radius,-radius,-radius,0,radius,radius,radius};
	int Ry[]={radius,radius,0,-radius,-radius,-radius,0,radius};
	uchar C;
    __int64 cnt = 0; 
	short*pf;
	for(int i=0;i<h;i++){//
		for (int j=0;j<w;j++){
			C = image[i+j*h];
			pf = ret+cnt*8;
			for(int k=0;k<8;k++){
				if(j+Ry[k]>=0 && j+Ry[k]<w && i+Rx[k]>=0 && i+Rx[k]<h)
					pf[k] = image[(i+Rx[k])+(j+Ry[k])*h]-C; 
				else
					pf[k] = 0;
			}
			cnt++;
		}
	}
}


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{ 	
    unsigned char* Image = (unsigned char*)mxGetPr(prhs[0]);//input image.
    int winsize = *mxGetPr(prhs[1]);//winsize
    int step = *mxGetPr(prhs[2]);//step
    int* L = (int*)mxGetPr(prhs[3]);//landmarks
    //get input struct.
    if(!mxIsStruct(prhs[4]))
    	mexErrMsgTxt("The 5-th parameter must be a model struct.");
    const char **fnames;       /* pointers to field names */
    mxArray    *tmp;
    int        ifield, nfields;
    mxClassID  *classIDflags;
    mwIndex    jstruct;
    int     nb_scales, nb_codes;
    /* get input arguments */
    nfields = mxGetNumberOfFields(prhs[4]);
    nb_scales = mxGetNumberOfElements(prhs[4]);
    /* allocate memory  for storing classIDflags */
    classIDflags = (mxClassID*)mxCalloc(nfields, sizeof(mxClassID));
    /* check empty field, proper data type, and data type consistency; and get classID for each field. */
    for(ifield=0; ifield<nfields; ifield++) { //for each field.
        for(jstruct = 0; jstruct < nb_scales; jstruct++) { //for each struct.
            tmp = mxGetFieldByNumber(prhs[4], jstruct, ifield);
            if(tmp == NULL) {
                mexPrintf("%s%d\t%s%d\n", "FIELD: ", ifield+1, "STRUCT INDEX :", jstruct+1);
                mexErrMsgIdAndTxt( "PRMI:LSP_test:fieldEmpty", "Above field is empty!");
            }
            if(jstruct==0) {
                if( (!mxIsDouble(tmp)) || mxIsInf(*mxGetPr(tmp)) || mxIsNaN(*mxGetPr(tmp)) || mxIsComplex(tmp)) {
                    mexPrintf("%s%d\t%s%d\n", "FIELD: ", ifield+1, "STRUCT INDEX :", jstruct+1);
                    mexErrMsgIdAndTxt( "PRMI:LSP_test:invalidField", "Above field must have double finite nonnan and noncomplex data.");
                }
                classIDflags[ifield]=mxGetClassID(tmp);
            } else {
                if (mxGetClassID(tmp) != classIDflags[ifield]) {
                    mexPrintf("%s%d\t%s%d\n", "FIELD: ", ifield+1, "STRUCT INDEX :", jstruct+1);
                    mexErrMsgIdAndTxt( "PRMI:LSP_test:invalidFieldType","Inconsistent data type in above field!");
                }
            }
            
        }
    }
    /* get field names*/
    const char *expected_fnames[] = {"tree","ncode","alpha","clft","cost","prob","radius"};
    if(nfields != 7){
    	mexPrintf("Input model has %d fields.\n",nfields);
    	mexErrMsgIdAndTxt( "PRMI:LSP_test:invalidField", "The encoding model must have 7 fields.");
    }
    fnames = (const char **)mxCalloc(nfields, sizeof(*fnames));
    for (ifield=0; ifield< nfields; ifield++){
        fnames[ifield] = mxGetFieldNameByNumber(prhs[4],ifield);
        if(strcmp(fnames[ifield],expected_fnames[ifield])!=0){
            mexPrintf("%s%d\t%s%d\texpected: %s, passed: %s\n", "FIELD: ", ifield+1, "STRUCT INDEX :", jstruct+1,expected_fnames[ifield],fnames[ifield]);
            mexErrMsgIdAndTxt( "PRMI:LSP_test:invalidFieldName","Inconsistent field name in above field!");
        }
    }
    /*Get field values*/
    double** trees = (double**)mxMalloc(sizeof(double*)*nb_scales);
    int* radii = (int*)mxMalloc(sizeof(int)*nb_scales);
    for (jstruct=0; jstruct<nb_scales; jstruct++) {
    	nb_codes = *mxGetPr(mxGetFieldByNumber(prhs[4],jstruct,1)); //ncode.
    	trees[jstruct] = (double*)mxMalloc(sizeof(double)*(2*nb_codes-1)*6);
    	memcpy(trees[jstruct],mxGetPr(mxGetFieldByNumber(prhs[4],jstruct,0)),sizeof(double)*(2*nb_codes-1)*6); //tree.
    	radii[jstruct] = *mxGetPr(mxGetFieldByNumber(prhs[4],jstruct,6)); //radius.
    }
    mxFree(classIDflags);
    mxFree(fnames);
    int height = mxGetM(prhs[0]);
    int width = mxGetN(prhs[0]);
    int nb_landmarks = mxGetM(prhs[3]);
	int* landmarks = 0;
	if (nb_landmarks == 0){ //need to generate landmarks.
		int numPatchX = width/step-1;
		int numPatchY = height/step-1;
		int xoff = (width%step)/2 + step;
		int yoff = (height%step)/2 + step;
		nb_landmarks = numPatchX*numPatchY;
		landmarks = (int*)mxMalloc(nb_landmarks*2*sizeof(int));
		get_fiducial_points_dense_sampling(width, height, landmarks, landmarks+nb_landmarks, nb_landmarks, step);
	}else{
		landmarks = (int*)mxMalloc(nb_landmarks*2*sizeof(int));
		for(int i = 0;i<nb_landmarks;i++) landmarks[i] = L[i*2];
		for(int i = 0;i<nb_landmarks;i++) landmarks[i+nb_landmarks] = L[i*2+1];
	}
    int nb_pf = height * width;
    short* pf = (short*)mxMalloc(nb_pf*8*sizeof(short));
    int *Icode1 = (int*)mxMalloc(sizeof(int)*height*width);
    int *Icode2 = 0;
    
    mxArray *features = mxCreateCellMatrix(1,nb_scales); //1st returned.
    if(nlhs>=2){ //2nd returned.
    	plhs[1] = mxCreateNumericMatrix(2, nb_landmarks, mxINT32_CLASS, mxREAL);
		int* ptr = (int*)mxGetPr(plhs[1]);
		for(int i = 0;i<nb_landmarks;i++) ptr[i*2] = landmarks[i];
		for(int i = 0;i<nb_landmarks;i++) ptr[i*2+1] = landmarks[i+nb_landmarks];
    }
    mxArray *encoded = 0;
    if(nlhs==3){
    	Icode2 = (int*)mxMalloc(sizeof(int)*height*width); //encoded image required to be returned.
    	encoded = mxCreateCellMatrix(1,nb_scales); //3rd returned.
    }
    int i,j,n,y,x,cnt;
    const int szPatchHalf = (1+winsize)/2;
    float* feat = (float*)mxMalloc(sizeof(float)*nb_landmarks*nb_codes); //each row is a local feature.
    for(i = 0;i<nb_scales;i++){ //for each scale.
		/*extract pixel feature*/
		extract_pixel_feat(pf, Image, height, width, radii[i]);    
		/*generate encoded image*/
		if(!Icode2) GetImageCode(pf, height, width, Icode1, trees[i], 2*nb_codes-1);
		else{
			GetImageCode(pf, height, width, Icode1, Icode2, trees[i], 2*nb_codes-1);
			plhs[2] = mxCreateNumericMatrix(height, width, mxINT32_CLASS,mxREAL);
		    memcpy(mxGetPr(plhs[2]),Icode2,sizeof(int)*height*width);
		    mxSetCell(encoded, i, mxDuplicateArray(plhs[2]));
		    mxDestroyArray(plhs[2]);
        }
		/*dense sampling to generate features*/
		const int* X = landmarks;
		const int* Y = landmarks + nb_landmarks;
        memset(feat, 0, sizeof(float)*nb_landmarks*nb_codes);
        cnt = 0;
        for (n = 0; n < nb_landmarks; n++){ //for each landmark point.
            for(y = Y[n]-szPatchHalf;y < Y[n]+szPatchHalf; y++){
                for(x = X[n]-szPatchHalf;x < X[n]+szPatchHalf; x++){
                    if(x>=0 && x<width && y>=0 && y<height)
                        feat[cnt+Icode1[y*width+x]]++;
				}
            }
			cnt = cnt + nb_codes;
        }
    	for(j= 0;j<nb_landmarks*nb_codes;j++)
    		feat[j] = log(1+feat[j]);
		plhs[0] = mxCreateNumericMatrix(nb_codes, nb_landmarks, mxSINGLE_CLASS, mxREAL);
		memcpy(mxGetPr(plhs[0]),feat,sizeof(float)*nb_codes*nb_landmarks);
		mxSetCell(features, i, mxDuplicateArray(plhs[0]));
		mxDestroyArray(plhs[0]);
    }
    plhs[0] = features;
    plhs[2] = encoded;
    for (jstruct=0; jstruct<nb_scales; jstruct++) mxFree(trees[jstruct]);
    mxFree(trees);
    mxFree(radii);
    mxFree(pf);
    mxFree(Icode1);
    if(Icode2) mxFree(Icode2);
    mxFree(feat);
    mxFree(landmarks);
}

