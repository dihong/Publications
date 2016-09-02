#include "mex.h" 
#include "stdlib.h"
#include "memory.h"
#include <string>
#include <vector>
#include <math.h>

using namespace std;

typedef unsigned char uchar;

#define LCHILD 0
#define RCHILD 1
#define ATB 2
#define TH 3
#define CODE 4
#define NSMPL 5

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
			Icode[i+j*height] = node[CODE];
			cnt++;
		}
	}
}

void extract_pixel_feat(short*ret, const uchar* image, int h, int w, int radius){
	int Rx[]={0,-radius,-radius,-radius,0,radius,radius,radius};
	int Ry[]={radius,radius,0,-radius,-radius,-radius,0,radius};
	uchar C;
    	unsigned long long int cnt = 0; 
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
    //get input struct.
    if(!mxIsStruct(prhs[1]))
    	mexErrMsgTxt("The 2-th parameter must be a model struct.");
    const char **fnames;       /* pointers to field names */
    mxArray    *tmp;
    int        ifield, nfields;
    mxClassID  *classIDflags;
    mwIndex    jstruct;
    int        nb_codes, nb_scales = 1;
    /* get input arguments */
    nfields = mxGetNumberOfFields(prhs[1]);
    /* allocate memory  for storing classIDflags */
    classIDflags = (mxClassID*)mxCalloc(nfields, sizeof(mxClassID));
    /* check empty field, proper data type, and data type consistency; and get classID for each field. */
    for(ifield=0; ifield<nfields; ifield++) { //for each field.
        for(jstruct = 0; jstruct < nb_scales; jstruct++) { //for each struct.
            tmp = mxGetFieldByNumber(prhs[1], jstruct, ifield);
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
        fnames[ifield] = mxGetFieldNameByNumber(prhs[1],ifield);
        if(strcmp(fnames[ifield],expected_fnames[ifield])!=0){
            mexPrintf("%s%d\t%s%d\texpected: %s, passed: %s\n", "FIELD: ", ifield+1, "STRUCT INDEX :", jstruct+1,expected_fnames[ifield],fnames[ifield]);
            mexErrMsgIdAndTxt( "PRMI:LSP_test:invalidFieldName","Inconsistent field name in above field!");
        }
    }
    /*Get field values*/
    double** trees = (double**)mxMalloc(sizeof(double*)*nb_scales);
    int* radii = (int*)mxMalloc(sizeof(int)*nb_scales);
    for (jstruct=0; jstruct<nb_scales; jstruct++) {
    	nb_codes = *mxGetPr(mxGetFieldByNumber(prhs[1],jstruct,1)); //ncode.
    	trees[jstruct] = (double*)mxMalloc(sizeof(double)*(2*nb_codes-1)*6);
    	memcpy(trees[jstruct],mxGetPr(mxGetFieldByNumber(prhs[1],jstruct,0)),sizeof(double)*(2*nb_codes-1)*6); //tree.
    	radii[jstruct] = *mxGetPr(mxGetFieldByNumber(prhs[1],jstruct,6)); //radius.
    }
    mxFree(classIDflags);
    mxFree(fnames);
    int height = mxGetM(prhs[0]);
    int width = mxGetN(prhs[0]);
    int nb_pf = height * width;
    short* pf = (short*)mxMalloc(nb_pf*8*sizeof(short));
    int *Icode = (int*)mxMalloc(sizeof(int)*height*width);
    extract_pixel_feat(pf, Image, height, width, radii[0]);
    GetImageCode(pf, height, width, Icode, trees[0], 2*nb_codes-1);
    plhs[0] = mxCreateNumericMatrix(height, width, mxINT32_CLASS,mxREAL);
    memcpy(mxGetPr(plhs[0]),Icode,sizeof(int)*height*width);

    for (jstruct=0; jstruct<nb_scales; jstruct++) mxFree(trees[jstruct]);
    mxFree(trees);
    mxFree(radii);
    mxFree(pf);
    mxFree(Icode);
}

