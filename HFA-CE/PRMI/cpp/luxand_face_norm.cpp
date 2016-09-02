#include <algorithm>
#include <memory.h>
#include <cmath>
#include <string>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <prmi/luxand/LuxandFaceSDK.h>
#include <fstream>

#ifdef __gnu_linux__
	#include <prmi/parallel.h>
#endif

typedef unsigned char uchar;

using namespace std;

const int hb = 135; //height of bounding box.
const int wb = 100; //width of bounding box.
const int npnt = 39; //number of landmark points.
const int ntri = 64; //number of triangles.
const int Rx[] = {0,-1,-1,-1,0,1,1,1};
const int Ry[] = {1,1,0,-1,-1,-1,0,1};

typedef struct FACE_DETECT_PARAM{
	vector<uchar*> faces; //face images, row major, same size, gray scale.
	vector<uchar*> norm_faces; //normalized faces
	int H1,W1; //input image dimension.
	int H2,W2; //output image dimension.
	double* mask; //template of mask.
	double* tshape_output; // output coordinate in the template.
} FACE_DETECT_PARAM;

#define pnt2tri(T,ind,X,Y,p1,p2,p3)\
T[ 6*ind ] = X[p1-1];\
        T[6*ind+1] = X[p2-1];\
        T[6*ind+2] = X[p3-1];\
                T[6*ind+3] = Y[p1-1];\
                T[6*ind+4] = Y[p2-1];\
                        T[6*ind+5] = Y[p3-1];\

double* get_tshape(double* shape){ //return ntri*6 matrix of the tshape.
    //puts("OUTLINE::set_tshape");
    double *X = new double [npnt],*Y = new double [npnt];
    for(int i = 0;i<npnt;i++){
    	X[i] = shape[2*i];
    	Y[i] = shape[2*i+1];
    }
    double* tshape = new double [ntri*6];
    double* T = tshape;
    pnt2tri(T,0,X,Y,11,15,37);
    pnt2tri(T,1,X,Y,11,15,23);
    pnt2tri(T,2,X,Y,11,18,23);
    pnt2tri(T,3,X,Y,4,11,18);
    pnt2tri(T,4,X,Y,18,22,23);
    pnt2tri(T,5,X,Y,19,22,23);
    pnt2tri(T,6,X,Y,4,18,28);
    pnt2tri(T,7,X,Y,18,22,28);
    pnt2tri(T,8,X,Y,19,22,26);
    pnt2tri(T,9,X,Y,12,19,23);
    pnt2tri(T,10,X,Y,12,15,23);
    
    pnt2tri(T,11,X,Y,22,26,28);
    pnt2tri(T,12,X,Y,1,17,26);
    pnt2tri(T,13,X,Y,1,17,27);
    pnt2tri(T,14,X,Y,1,26,28);
    pnt2tri(T,15,X,Y,1,27,29);
    pnt2tri(T,16,X,Y,1,28,30);
    pnt2tri(T,17,X,Y,1,29,30);
    pnt2tri(T,18,X,Y,28,30,33);
    pnt2tri(T,19,X,Y,29,30,34);
    pnt2tri(T,20,X,Y,30,31,33);
    
    pnt2tri(T,21,X,Y,30,31,34);
    pnt2tri(T,22,X,Y,2,33,35);
    pnt2tri(T,23,X,Y,32,33,35);
    pnt2tri(T,24,X,Y,31,32,33);
    pnt2tri(T,25,X,Y,31,32,34);
    pnt2tri(T,26,X,Y,32,34,36);
    pnt2tri(T,27,X,Y,3,34,36);
    pnt2tri(T,28,X,Y,2,28,33);
    pnt2tri(T,29,X,Y,3,29,34);
    pnt2tri(T,30,X,Y,2,4,28);
    
    pnt2tri(T,31,X,Y,2,4,6);
    pnt2tri(T,32,X,Y,2,6,35);
    pnt2tri(T,33,X,Y,6,8,35);
    pnt2tri(T,34,X,Y,8,32,35);
    pnt2tri(T,35,X,Y,8,10,32);
    pnt2tri(T,36,X,Y,9,10,32);
    pnt2tri(T,37,X,Y,9,32,36);
    pnt2tri(T,38,X,Y,7,9,36);
    pnt2tri(T,39,X,Y,3,7,36);
    pnt2tri(T,40,X,Y,3,5,7);
    
    pnt2tri(T,41,X,Y,3,5,29);
    pnt2tri(T,42,X,Y,5,21,29);
    pnt2tri(T,43,X,Y,24,27,29);
    pnt2tri(T,44,X,Y,21,24,29);
    pnt2tri(T,45,X,Y,5,14,21);
    pnt2tri(T,46,X,Y,17,20,27);
    pnt2tri(T,47,X,Y,20,24,27);
    pnt2tri(T,48,X,Y,12,13,17);
    pnt2tri(T,49,X,Y,13,17,20);
    pnt2tri(T,50,X,Y,13,20,25);
    
    pnt2tri(T,51,X,Y,20,24,25);
    pnt2tri(T,52,X,Y,21,24,25);
    pnt2tri(T,53,X,Y,13,16,25);
    pnt2tri(T,54,X,Y,14,16,25);
    pnt2tri(T,55,X,Y,14,21,25);
    pnt2tri(T,56,X,Y,14,16,39);
    pnt2tri(T,57,X,Y,16,38,39);
    pnt2tri(T,58,X,Y,13,16,38);
    pnt2tri(T,59,X,Y,12,13,38);
    pnt2tri(T,60,X,Y,12,15,38);

    pnt2tri(T,61,X,Y,15,37,38);
	pnt2tri(T,62,X,Y,17,19,26);
	pnt2tri(T,63,X,Y,12,17,19);

	delete X,Y;
	
    return T;
}

//=======================================================================
//This function computes the (r,B) associated to the tshape.
void computeBr(double* r, double* B, double* tshape){ //ok.
	double d;
	int k;
	for(int i=0;i<ntri;i++){
		k = i*6;
		d = (tshape[k+1]-tshape[k])*(tshape[k+5]-tshape[k+3])-(tshape[k+2]-tshape[k])*(tshape[k+4]-tshape[k+3]);
		r[i*3] = (tshape[k+1]*tshape[k+5]-tshape[k+2]*tshape[k+4])/d;//q1
		r[i*3+1] = (tshape[k+2]*tshape[k+3]-tshape[k]*tshape[k+5])/d;//q2
		r[i*3+2] = (tshape[k]*tshape[k+4]-tshape[k+1]*tshape[k+3])/d;//q3
		B[k] = (tshape[k+2]-tshape[k+1])/d;//X1
		B[k+1] = (tshape[k]-tshape[k+2])/d;//X2
		B[k+2] = (tshape[k+1]-tshape[k])/d;//X3
		B[k+3] = (tshape[k+4]-tshape[k+5])/d;//Y1
		B[k+4] = (tshape[k+5]-tshape[k+3])/d;//Y2 
		B[k+5] = (tshape[k+3]-tshape[k+4])/d;//Y3
	}
}

//========================================================================
//warp the frame from tshape_input into the tshape_output, and return the warped face within the bounding box.
unsigned char* Warp(unsigned char* frame, int h, int w, double* tshape_input, double* tshape_output, double* mask){
    unsigned char* face = new unsigned char [wb*hb];
    double* rr = new double [ntri*3];
    double* BB = new double [ntri*6];
    computeBr(rr,BB,tshape_output);
	int i,j,k;
	double* T = new double [ntri*6];	
	memset(T,0,sizeof(double)*ntri*6);
	//compute T
	for(j=0;j<3;j++){ //for each point of the triangle.
		k = j+3;
		for(i=0;i<ntri*6;i+=6){ //for each triangle.
			T[i] += rr[i/2+j]*tshape_input[i+j];//C1.
			T[i+1] += rr[i/2+j]*tshape_input[i+k];//C2.
			T[i+2] += BB[i+k]*tshape_input[i+j]; //A1.
			T[i+3] += BB[i+j]*tshape_input[i+j]; //A2.
			T[i+4] += BB[i+k]*tshape_input[i+k]; //A3.
			T[i+5] += BB[i+j]*tshape_input[i+k]; //A4.
		}
	}
    
	int t;
	double * pT = 0;
    for(i=0;i<wb*hb;i++)
		face[i] = 255;
    double u,v,xf,yf;
    int xi,yi;
    for(i=0;i<wb;i++){ //for each point in the mask coordinate.
        for(j=0;j<hb;j++){
            t = mask[i+j*wb] - 1; //read the triangle index of the current point: 1-ntri, 255.
            if(t>=0 && mask[i+(j+2)*wb] && mask[i+(j-2)*wb]&& mask[(i+2)+j*wb] && mask[(i-2)+j*wb]){ //within the facial area.
                pT = T+t*6; //the start point of the affine warp params for the t-th triangle.
                xf = pT[0]+pT[2]*j+pT[3]*i;
                yf = pT[1]+pT[4]*j+pT[5]*i;
                xi = xf; yi = yf; //floor.
                u = xf-xi; v = yf-yi;
                if(yi+1<w && xi+1<h && xi >=0 && yi >= 0)
					face[j*wb+i] = (1-v)*(1-u)*frame[xi*w+yi]+v*(1-u)*frame[xi*w+(yi+1)]+(1-v)*u*frame[(xi+1)*w+yi]+v*u*frame[(xi+1)*w+(yi+1)];
            }
        }
    }
    delete T, rr, BB;
	return face;
}



bool detect_facial_features(HImage& imageHandle,double*y)
{
    TFacePosition facePosition;
    FSDK_Features facialFeatures;
    if (FSDK_DetectFace(imageHandle, &facePosition) == FSDKE_OK ){
        if(FSDK_DetectFacialFeaturesInRegion(imageHandle, &facePosition, &facialFeatures)!=FSDKE_OK){
            FSDK_FreeImage(imageHandle);
            return false;
        }
    }
    else{
        FSDK_FreeImage(imageHandle);
        return false;
    }
    int cnt = 0;
    for (int i = 0; i < FSDK_FACIAL_FEATURE_COUNT; i++)
    {
        if((i>=2 && i<=17) || (i>=22 && i<=28) || (i>=31 && i<=32) || (i>=43 && i<=46) || (i>=54 && i<=59) || (i==49)){
            y[2*cnt] = facialFeatures[i].y;
            y[2*cnt+1] = facialFeatures[i].x;
            cnt++;
        }
    }
    return true;
}

void thread_face_detection(void* param) {
    FACE_DETECT_PARAM& p = *(FACE_DETECT_PARAM*) param;
    p.H2 = hb;
    p.W2 = wb;
	double eyeL[2],eyeR[2],X[6][2],Xi[2][6],XtX[2][2],XtXi[2][2],Y[6],beta,d,e[2],features[39][2],det;
	for(int i = 0;i<p.faces.size();i++){ //for each face image.
		// load image.
		HImage image;
		if(FSDKE_OK != FSDK_LoadImageFromBuffer(&image, p.faces[i], p.W1, p.H1, p.W1,FSDK_IMAGE_GRAYSCALE_8BIT)){
			p.norm_faces.push_back(0);
			continue;
		}
		//detect features.
		if(!detect_facial_features(image,&features[0][0])){
			p.norm_faces.push_back(0);
			FSDK_FreeImage(image);
			continue;
		}
		else{
			features[23][0] = features[23][0] + (features[23][0]-features[24][0])*0.2;
			//set X[6][2]
			for(int j=0;j<6;j++)
				X[j][0] = 1.0;
			X[0][1] = features[0][0];
			X[1][1] = features[9][0];
			X[2][1] = features[16][0];
			X[3][1] = features[29][0];
			X[4][1] = features[30][0];
			X[5][1] = features[31][0];
			//set Y[6]
			Y[0] = features[0][1];
			Y[1] = features[9][1];
			Y[2] = features[16][1];
			Y[3] = features[29][1];
			Y[4] = features[30][1];
			Y[5] = features[31][1];
			//compute (X'*X)\X'
			memset(XtX,0,sizeof(double)*4);
			for(int j = 0;j<6;j++) XtX[0][0] += X[j][0]*X[j][0];
			for(int j = 0;j<6;j++) XtX[1][1] += X[j][1]*X[j][1];
			for(int j = 0;j<6;j++) XtX[0][1] += X[j][0]*X[j][1];
			XtX[1][0] = XtX[0][1];
			det = XtX[0][0]*XtX[1][1]-XtX[0][1]*XtX[1][0];
			XtXi[0][0] = XtX[1][1]/det;
			XtXi[1][1] = XtX[0][0]/det;
			XtXi[0][1] = -XtX[0][1]/det;
			XtXi[1][0] = XtXi[0][1];
			memset(Xi,0,sizeof(double)*12);
			for(int j = 0;j<6;j++) Xi[0][j] = XtXi[0][0]*X[j][0]+XtXi[0][1]*X[j][1];
			for(int j = 0;j<6;j++) Xi[1][j] = XtXi[1][0]*X[j][0]+XtXi[1][1]*X[j][1];
			//compute beta.
			beta = 0;
			for(int j = 0;j<6;j++) beta += Xi[1][j]*Y[j];
		    //compute d,e
		    d = sqrt(pow(features[16][0]-features[30][0],2)+pow(features[16][1]-features[30][1],2));
		    e[0] = cos(atan(beta))*d;
		    e[1] = sin(atan(beta))*d;
		    //compute extended features.
		    features[36][0] = features[10][0] - 0.8*e[0];
		    features[36][1] = features[10][1] - 0.8*e[1];
		    features[37][0] = features[16][0] - e[0];
		    features[37][1] = features[16][1] - e[1];
		    features[38][0] = features[13][0] - 0.8*e[0];
		    features[38][1] = features[13][1] - 0.8*e[1];
			//warp the image.
			unsigned char*Buffer = new unsigned char [p.W1*p.H1];
			if(FSDKE_OK != FSDK_SaveImageToBuffer(image, Buffer, FSDK_IMAGE_GRAYSCALE_8BIT)){
				p.norm_faces.push_back(0);
				FSDK_FreeImage(image);
				continue;
			}
			double* tshape = get_tshape(&features[0][0]);
			unsigned char* warped = Warp(Buffer, p.H1, p.W1, tshape, p.tshape_output, p.mask);
			p.norm_faces.push_back(warped);
			FSDK_FreeImage(image);
			delete Buffer,tshape;
		}
	}
}

/*  This function normalizes face images with Luxand face detector.
*	Return values:
*	0 -- successful
*	1 -- activation file not found.
*	2 -- activation failed.
*	3 -- could not open mask.txt.
*	4 -- could not open frontal_135x100.dat.
*	5 -- no input face.
*/
int luxand_face_norm(const vector<uchar*>& faces,const int & H1, const int & W1, vector<uchar*>& norm_faces, int& H2, int& W2)
{
	H2 = -1;
	W2 = -1;
	//FSDK initialization.
    string key;
    ifstream is("/usr/local/PRMI/lic/luxand/license.lic"); ///usr/local/Menily/
    if (!is.is_open())
    	return 1;
    getline(is,key);
    if (FSDKE_OK != FSDK_ActivateLibrary(key.c_str()))
    	return 2;
    is.close();
    FSDK_Initialize(0);
    FSDK_SetFaceDetectionParameters(true, true, 256);
    FSDK_SetFaceDetectionThreshold(3);
	
    //load mask and pnts.
    double* mask = new double [hb*wb];
    double* pnts = new double [npnt*2];
    int tmp,ret;
    FILE* fp = fopen("/usr/local/PRMI/model/face_norm/mask.txt","r");
    if(!fp)
    	return 3;
    for(int i = 0;i<hb;i++)
    	for(int j = 0;j<wb;j++){
    		ret = fscanf(fp,"%d",&tmp);
    		mask[i*wb+j] = tmp;
    	}
    fclose(fp);
    fp = fopen("/usr/local/PRMI/model/face_norm/frontal_135x100.dat","r");
    if(!fp)
    	return 4;
    for(int i = 0;i<npnt;i++){
    	ret = fscanf(fp,"%d",&tmp); pnts[i*2] = tmp;
    	ret = fscanf(fp,"%d",&tmp); pnts[i*2+1] = tmp;
    }
    fclose(fp);
    
    double* tshape_output = get_tshape(pnts);
    delete pnts;
    
    // set up parallel threads.
#ifdef __gnu_linux__
	Parallel PL;
#endif
	int nb_faces = faces.size();
	if(nb_faces==0) return 5;
	vector<int> nb_faces_per_thread;
	int nb_threads = 12;
	if(nb_threads>nb_faces){
		nb_threads = nb_faces;
		for(int i = 0;i<nb_threads;i++)
			nb_faces_per_thread.push_back(1);
	}else{
		int cnt = 0;
		for(int i=0;i<nb_threads;i++){
			nb_faces_per_thread.push_back(1);
			cnt++;
		}
		while(cnt<nb_faces){
			for(int i=0;i<nb_threads && cnt<nb_faces;i++){
				nb_faces_per_thread[i]++;
				cnt++;
			}
		}
	}
    vector<void*> thread_params;
    FACE_DETECT_PARAM* params = new FACE_DETECT_PARAM [nb_threads];
    int cnt = 0;
    for (int k = 0; k < nb_threads; k++){
        params[k].tshape_output = tshape_output;
        params[k].mask = mask;
        params[k].H1 = H1;
        params[k].W1 = W1;
        for(int i=0;i<nb_faces_per_thread[k];i++)
        	params[k].faces.push_back(faces[cnt++]);
        thread_params.push_back(&params[k]);
    }
    
#ifdef __gnu_linux__
    PL.Run(thread_face_detection,thread_params);
#else
	for(int i=0;i<nb_faces;i++) thread_face_detection(thread_params[i]);
#endif
	
	H2 = params[0].H2;
	W2 = params[0].W2;
	
	for (int k = 0; k < nb_threads; k++){
		for(int i = 0;i<params[k].faces.size();i++){
			norm_faces.push_back(params[k].norm_faces[i]);
		}
	}
    delete mask,pnts;
    FSDK_Finalize();
    return 0;
}
