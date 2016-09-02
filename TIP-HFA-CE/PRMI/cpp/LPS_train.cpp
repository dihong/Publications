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
#define INF 100000

int mode = 0;
int aligned = 0;

int luxand_face_norm(const vector<uchar*>& faces,const int & H1, const int & W1, vector<uchar*>& norm_faces, int& H2, int& W2);

//the struct of encoding tree.
class ENC_TREE{
	public:
	double* table; //[total_nb_nodes*6]: 
	/* left: 0 to total_nb_nodes-1, integer, the index of left child of current node.
	 * right: 0 to total_nb_nodes-1, integer, the index of right child of current node.
	 * atb: best split attribute that maximizes the entropy.
	 * th: best split threshold that maximizes the entropy [obtained by brute-force search].
	 * code: the code assigned to this partition (internal nodes are -1, leaf nodes are greater or equal to 0)
     * nsmpl: number of samples fall into this partition.
     */
	const int nb_codes; //number of leaf nodes in the tree in total. 
	const int total_nb_nodes;
	int current_nb_nodes;
	const int nb_elem; //6
	const __int64 nb_pixel_pairs; //number of training pixel pairs.
	__int64 nb_conflicting_pixel_pairs; //number of training pixel pairs that are assigned to different partitions in the encoding tree.
	double alpha; //the balancing factor: cost = alpha*nb_conflicting_pixel_pairs/nb_pixel_pairs+(1-alpha)*()/log();
	ENC_TREE(int ncodes, __int64 nb_px_pairs, double a):nb_codes(ncodes),nb_elem(6),total_nb_nodes(2*ncodes-1),nb_pixel_pairs(nb_px_pairs){
		if(nb_codes<=0)
			mexErrMsgTxt("The number of codes must be positive integer.");
        table = (double*)mxMalloc(sizeof(double)*total_nb_nodes*nb_elem);
        memset(table,0,sizeof(double)*total_nb_nodes*nb_elem);
		current_nb_nodes = 0;
		nb_conflicting_pixel_pairs = 0;
		alpha = a;
		if(alpha<=0 || alpha>=1)
			mexErrMsgTxt("The alpha must be a fraction number between (0,1).");
	}
	~ENC_TREE(){
        mxFree(table);
	}
};

typedef struct NODE{
	__int64 *idx1, *idx2; //samples hold by this node.
	__int64 N1,N2; //number of samples hold by this node.
	double* self; //the row of table in ENC_TREE
	double* left; //the row of table in ENC_TREE
	double* right; //the row of table in ENC_TREE
    int id; //node id
    int atb; // best attribute to expand this node, -1 if not evaluated.
    float th;
    float cost; //cost of expanding the node using this attribute.
	__int64 left_N1,left_N2,right_N1,right_N2; //number of samples hold by this node.
	__int64 num_conflicting_pairs;
}NODE;

typedef struct PARALLEL_THREAD_EVAL_COST{
    ENC_TREE* tree;
    NODE* node;
    short* pf1, *pf2;
    int atb;
    float th;
    float cost; //cost of expanding the node using this attribute.
	__int64 left_N1,left_N2,right_N1,right_N2; //number of samples hold by this node.
    __int64 num_conflicting_pairs;
    short* F1;
    short* F2;
    int begin,end,step; //the beging/end/step searching points of th.
}PARALLEL_THREAD_EVAL_COST;

void align_faces(vector<uchar*>& images1, vector<uchar*>& images2, int& h1, int& w1){
	if(images1.size() != images2.size())
		mexErrMsgTxt("image1 & image2 must have the same number of faces.");
	//convert col-major to row-major
	uchar* buffer = (uchar*)mxMalloc(h1*w1);
	int i,j,k,n;
	for(i = 0;i<images1.size();i++){
		for(j = 0;j<h1;j++)
			for(k = 0;k<w1;k++)
				buffer[j*w1+k] = images1[i][k*h1+j];
		memcpy(images1[i],buffer,h1*w1);
	}
	for(i = 0;i<images2.size();i++){
		for(j = 0;j<h1;j++)
			for(k = 0;k<w1;k++)
				buffer[j*w1+k] = images2[i][k*h1+j];
		memcpy(images2[i],buffer,h1*w1);
	}
	vector<uchar*> norm_faces_1,norm_faces_2;
	int h2,w2; //dimension of normalized faces.
	int c1 = luxand_face_norm(images1,h1,w1,norm_faces_1, h2, w2);
	int c2 = luxand_face_norm(images2,h1,w1,norm_faces_2, h2, w2);
	if(c1 || c2){
		mexPrintf("LSP_train.cpp: Failed to call luxand_face_norm function: error code is %d.",c1);
		mexErrMsgTxt("Error.");
	}
	//convert row-major to col-major
	for(i = 0;i<images1.size();i++){
		for(j = 0;j<h1;j++)
			for(k = 0;k<w1;k++)
				buffer[k*h1+j] = images1[i][j*w1+k];
		memcpy(images1[i],buffer,h1*w1);
	}
	for(i = 0;i<images2.size();i++){
		for(j = 0;j<h1;j++)
			for(k = 0;k<w1;k++)
				buffer[k*h1+j] = images2[i][j*w1+k];
		memcpy(images2[i],buffer,h1*w1);
	}
	mxFree(buffer);
	//
	for(i = 0;i<images1.size();i++){
		if(norm_faces_1[i] && norm_faces_2[i]){
			images1[i] = norm_faces_1[i];
			images2[i] = norm_faces_2[i];
		}else{
			images1[i] = 0;
			images2[i] = 0;
			if(norm_faces_1[i]) delete norm_faces_1[i];
			if(norm_faces_2[i]) delete norm_faces_2[i];
		}
	}
	h1 = h2;
	w1 = w2;
}

//extract pixel features with alignment.
/*__int64 extract_pixel_feat(short* ret1, short* ret2, const vector<uchar*>& images1, const vector<uchar*>& images2, int h, int w, int radius){
	if(images1.size() != images2.size()) return 0;
	
	int Rx[]={0,-radius,-radius,-radius,0,radius,radius,radius};
	int Ry[]={radius,radius,0,-radius,-radius,-radius,0,radius};
	
	//convert col-major to row-major
	uchar* buffer = (uchar*)mxMalloc(h*w);
	int i,j,k,n;
	for(i = 0;i<images1.size();i++){
		for(j = 0;j<h;j++)
			for(k = 0;k<w;k++)
				buffer[j*w+k] = images1[i][k*h+j];
		memcpy(images1[i],buffer,h*w);
	}
	for(i = 0;i<images2.size();i++){
		for(j = 0;j<h;j++)
			for(k = 0;k<w;k++)
				buffer[j*w+k] = images2[i][k*h+j];
		memcpy(images2[i],buffer,h*w);
	}
	mxFree(buffer);
	vector<uchar*> norm_faces_1,norm_faces_2;
	int h2,w2; //dimension of normalized faces.
	int c1 = luxand_face_norm(images1,h,w,norm_faces_1, h2, w2);
	int c2 = luxand_face_norm(images2,h,w,norm_faces_2, h2, w2);
	if(c1 || c2){
		mexPrintf("LSP_train.cpp: Failed to call luxand_face_norm function: error code is %d.",c1);
		mexErrMsgTxt("Error.");
	}
	__int64 cnt = 0;
	int cnt1 = 0, cnt2 = 0, cnt0 = 0;
	uchar *data1, *data2;
	short *pf, C;
	
	for(n = 0;n<norm_faces_1.size();n++){
		if(norm_faces_1[n]) cnt1++;
		if(norm_faces_2[n]) cnt2++;
		if(norm_faces_1[n] && norm_faces_2[n]){
			cnt0++;
			data1 = norm_faces_1[n];
			data2 = norm_faces_2[n];
			for(i = .2*h; i<.8*h;i++){
				for(j = .2*w;j<.8*w;j++){
					C = data1[i*w+j];
					pf = ret1+cnt*8;
					for(k=0;k<8;k++)
						pf[k] = data1[(i+Rx[k])*w+(j+Ry[k])]-C; 
						
					C = data2[i*w+j];
					pf = ret2+cnt*8;
					for(k=0;k<8;k++)
						pf[k] = data2[(i+Rx[k])*w+(j+Ry[k])]-C; 
					cnt++;
				}
			}
		}
	}
    if(mode){
		mexPrintf("Num Detected faces1 is %d, Num Detected faces2 is %d, Total num of faces used for training = %d.\n",cnt1,cnt2,cnt0);
		mexEvalString("drawnow;"); // to dump string.
    }
	return cnt;
}*/

//extract pixel features without alignment.
__int64 extract_pixel_feat(short*ret, const vector<uchar*>& images, int h, int w, int radius){
	int Rx[]={0,-radius,-radius,-radius,0,radius,radius,radius};
	int Ry[]={radius,radius,0,-radius,-radius,-radius,0,radius};
    int N = images.size();
	short C;
    __int64 cnt = 0; 
	short*pf; uchar* data = 0;
	for(int n=0;n<N;n++){
		data = images[n];
		if(data==0) continue;
		for(int i=.2*h;i<.8*h;i++){//
			for (int j=.2*w;j<.8*w;j++){
				C = data[j*h+i];
				pf = ret+cnt*8;
				for(int k=0;k<8;k++){
					pf[k] = data[(i+Rx[k])+(j+Ry[k])*h]-C; 
				}
				cnt++;
			}
		}
	}
	return cnt;
}

void parallel_thread_eval_cost(void* par){
    PARALLEL_THREAD_EVAL_COST& params = *(PARALLEL_THREAD_EVAL_COST*)par;
    NODE& node = *params.node;
    ENC_TREE& tree = *params.tree;
    __int64* L1 = node.idx1;
    __int64* L2 = node.idx2;
    short* pf1 = params.pf1;
    short* pf2 = params.pf2;
    if(node.N1<=0 || node.N2<=0){
        params.cost = INF; //invalid.
        return;
    }
    short* F1 = params.F1;
    short* F2 = params.F2;
    
    const int atb = params.atb;
    double mean = 0, std = 0;
    for(__int64 i = 0;i<node.N1;i++){
        F1[i] = pf1[8*L1[i]+atb];
        mean += F1[i];
    }
    for(__int64 i = 0;i<node.N2;i++){
        F2[i] = pf2[8*L2[i]+atb];
        mean += F2[i];
    }
    mean /= (node.N1+node.N2);
    for(__int64 i = 0;i<node.N1;i++)
    	std += pow(F1[i]-mean,2);
    for(__int64 i = 0;i<node.N2;i++)
    	std += pow(F2[i]-mean,2);
    std = sqrt(std/(node.N1+node.N2));
    const int N = 50;
    int step = std/N; if(step<1) step = 1;
    float best_cost = INF;
    int best_th = -1;
    __int64 best_num_left_1 = 0, best_num_left_2 = 0, best_num_right_1 = 0, best_num_right_2 = 0;
    int begin = mean-0.5*std;
    int end = mean+0.5*std;
    
    short min1 = F1[0], max1 = F1[0], min2 = F2[0], max2 = F2[0];
    for(__int64 i = 1;i<node.N1;i++)
        if(F1[i]<min1) min1 = F1[i];
        else if(F1[i]>max1) max1 = F1[i];
    for(__int64 i = 1;i<node.N2;i++)
        if(F2[i]<min2) min2 = F2[i];
        else if(F2[i]>max2) max2 = F2[i];
    
    if(begin<min(min1,min2)) begin = min(min1,min2);  
    if(end>max(max1,max2)) end = max(max1,max2);  
    
    for(int th = begin; th<=end; th+=step){
        __int64 pL1 = 0, pL2 = 0; //point to the matching pair.
        __int64 num_left_1 = 0, num_left_2 = 0, num_right_1 = 0, num_right_2 = 0;
        params.num_conflicting_pairs = 0;
        while(pL1<node.N1 || pL2<node.N2){
            while((pL2>=node.N2 || L1[pL1]<L2[pL2]) && pL1<node.N1){
                if(F1[pL1]<th) num_left_1++;
                else num_right_1++;
                pL1++;
            }
            while((pL1>=node.N1 || L2[pL2]<L1[pL1]) && pL2<node.N2){
                if(F2[pL2]<th) num_left_2++;
                else num_right_2++;
                pL2++;
            }
            if(pL1<node.N1 && pL2<node.N2 && L1[pL1] == L2[pL2]){ //this is a matching pair.
                if(F1[pL1]<th) num_left_1++;
                else num_right_1++;
                if(F2[pL2]<th) num_left_2++;
                else num_right_2++;
                if(F1[pL1]<th && F2[pL2]>=th || F1[pL1]>=th && F2[pL2]<th) //this matching pair is assigned to two different partitions.
                    params.num_conflicting_pairs++;
                pL1++;
                pL2++;
            }
        }
        //compute the cost of this th.
        float p1 = (num_left_1+num_left_2)/(float)tree.nb_pixel_pairs;
        float p2 = (num_right_1+num_right_2)/(float)tree.nb_pixel_pairs;
        float cost;
    	if(p1==0 || p2==0)
    		cost = INF;
        else
        	cost = tree.alpha*params.num_conflicting_pairs/tree.nb_pixel_pairs + (1-tree.alpha)*(p1*log(p1)+p2*log(p2)-(p1+p2)*log(p1+p2))/log(tree.nb_codes);
        //
        if(cost<best_cost){
            best_cost = cost;
            best_th = th;
            best_num_left_1 = num_left_1, best_num_left_2 = num_left_2, best_num_right_1 = num_right_1, best_num_right_2 = num_right_2;
        }
    }
    //return results.
    params.cost = best_cost;
    params.th = best_th;
    params.left_N1 = best_num_left_1;
    params.left_N2 = best_num_left_2;
    params.right_N1 = best_num_right_1;
    params.right_N2 = best_num_right_2;
    params.begin = begin;
    params.end = end;
    params.step = step;
}

void train_encode_tree(ENC_TREE& tree, short* pf1, short*pf2, int ncode){
    NODE* tree_nodes = (NODE*)mxMalloc(tree.total_nb_nodes*sizeof(NODE));
    for(int i = 0;i<tree.total_nb_nodes;i++) tree_nodes[i].id = i;
    NODE& root = tree_nodes[0];
    __int64 num_pf = tree.nb_pixel_pairs;
    root.idx1 = (__int64*)mxMalloc(num_pf*sizeof(__int64));
    root.idx2 = (__int64*)mxMalloc(num_pf*sizeof(__int64));
    root.N1 = num_pf;
    root.N2 = num_pf;
    root.self = tree.table+tree.current_nb_nodes*tree.nb_elem;
    root.self[CODE] = 0;
    root.self[NSMPL] = root.N1 + root.N2;
    root.left = 0;
    root.right = 0;
    root.atb = -1;// not evaluated.
    int* code_pf1 = (int*)mxMalloc(sizeof(int)*num_pf);
    int* code_pf2 = (int*)mxMalloc(sizeof(int)*num_pf);
	for(__int64 i = 0;i<num_pf;i++){
		root.idx1[i] = i;
        root.idx2[i] = i;
        code_pf1[i] = 0;
        code_pf2[i] = 0;
    }
    
    //begin growing the tree.
    tree.current_nb_nodes = 1;
    PARALLEL_THREAD_EVAL_COST* params = (PARALLEL_THREAD_EVAL_COST*)mxMalloc(8*sizeof(PARALLEL_THREAD_EVAL_COST));
    
    for(int it = 1;it<tree.nb_codes;it++){
        NODE* best_node = 0;
        //evaluate the cost of expanding a leaf node.
        for(int i = 0;i<tree.current_nb_nodes;i++){
            if(tree.table[i*tree.nb_elem+CODE] != -1){ //for each leaf node.
                NODE& node = tree_nodes[i];
                if(node.atb>=0){
                    if(best_node == 0 || node.cost< best_node->cost)
                        best_node = &node;
                    continue;
                }
                __int64* L1 = node.idx1;
                __int64* L2 = node.idx2;
                float min_cost = -1;
                int best_atb = -1;
                float best_th = 0;
                vector<void*> thread_params;
                for(int j = 0;j<8;j++){
                    params[j].tree = &tree;
                    params[j].node = tree_nodes + i;
                    params[j].pf1 = pf1;
                    params[j].pf2 = pf2;
                    params[j].atb = j;
                    params[j].F1 = (short*)mxMalloc(node.N1*sizeof(short));
    				params[j].F2 = (short*)mxMalloc(node.N2*sizeof(short));
                    thread_params.push_back(params+j);
                }
                if(mode){
		            mexPrintf("-----------------------------------------------\n");
		            mexPrintf("nid = %d, N1 = %d, N2 = %d\n",node.id,node.N1,node.N2);
		            mexPrintf("Expanding node %d ... ",it);
		            mexEvalString("drawnow;"); // to dump string.
                }
#ifdef __gnu_linux__
                Parallel PL;
                PL.Run(parallel_thread_eval_cost,thread_params);
#else
                for (int j = 0; j < 8; j++)
                    parallel_thread_eval_cost(thread_params[j]);
#endif
                if(mode) mexPrintf("done.\n");
                //select the best attribute that minimizes the cost.
                PARALLEL_THREAD_EVAL_COST* best_param = 0;
                for (int j = 0; j < 8; j++){
                    if(best_param==0 || params[j].cost < best_param->cost)
                        best_param = &params [j];
                    double c = (double)params[j].num_conflicting_pairs/(node.N1+node.N2);
                    double l = (double)(params[j].left_N1+params[j].left_N2)/(node.N1+node.N2);
                    double r = (double)(params[j].right_N1+params[j].right_N2)/(node.N1+node.N2);
                    mxFree(params[j].F1);
                    mxFree(params[j].F2);
                    if(mode){
		                mexPrintf("atb = %d, th = %.4f, cost = %.4f, begin = %d, end = %d, step = %d, |C| = %.4f, |L| = %.4f, |R| = %.4f\n",params[j].atb,params[j].th,params[j].cost,params[j].begin,params[j].end,params[j].step,c,l,r);
		                mexEvalString("drawnow;"); // to dump string.
                    }
                }
                //set the evaluation result for this node.
                node.atb = best_param->atb;
                node.th = best_param->th;
                node.cost = best_param->cost;
                node.left_N1 = best_param->left_N1;
                node.left_N2 = best_param->left_N2;
                node.right_N1 = best_param->right_N1;
                node.right_N2 = best_param->right_N2;
                //if this is the current best_node, save it.
                if(best_node==0 || node.cost < best_node->cost)
                    best_node = &node;
            }
        }
        if(best_node->cost==INF){
        	mexPrintf("[Error] The maximum number of partitions is %d\n",it);
        	mexErrMsgTxt("No more node can be expanded.");
        }
        if(mode){
		    mexPrintf("*Expanded node %d at atb = %d with cost = %.10f\n",best_node->id,best_node->atb,best_node->cost);
		    mexEvalString("drawnow;"); // to dump string.
        }
        //expand a leaf node that minimizes the cost.
        int nid_left = tree.current_nb_nodes;
        int nid_right = tree.current_nb_nodes +1;
        NODE& lchild = tree_nodes[nid_left];
        NODE& rchild = tree_nodes[nid_right];
        
        best_node->left  = tree.table+nid_left *tree.nb_elem;
        best_node->right = tree.table+nid_right*tree.nb_elem;
        best_node->self[LCHILD] = nid_left;
        best_node->self[RCHILD] = nid_right;
        best_node->self[ATB] = best_node->atb;
        best_node->self[TH] = best_node->th;
        best_node->self[CODE] = -1; //it is now become internal node.

        lchild.idx1 = (__int64*)mxMalloc(best_node->left_N1*sizeof(__int64));
        lchild.idx2 = (__int64*)mxMalloc(best_node->left_N2*sizeof(__int64));
        lchild.N1 = best_node->left_N1;
        lchild.N2 = best_node->left_N2;
        lchild.self = best_node->left;
        lchild.self[CODE] = 0;
        lchild.self[NSMPL] = lchild.N1 + lchild.N2;
        lchild.left = 0;
        lchild.right = 0;
        lchild.atb = -1;
        
        rchild.idx1 = (__int64*)mxMalloc(best_node->right_N1*sizeof(__int64));
        rchild.idx2 = (__int64*)mxMalloc(best_node->right_N2*sizeof(__int64));
        rchild.N1 = best_node->right_N1;
        rchild.N2 = best_node->right_N2;
        rchild.self = best_node->right;
        rchild.self[CODE] = 0;
        rchild.self[NSMPL] = rchild.N1 + rchild.N2;
        rchild.left = 0;
        rchild.right = 0;
        rchild.atb = -1;
        __int64 cnt_left = 0, cnt_right = 0;
        for(__int64 i = 0;i<best_node->N1;i++){
            if(pf1[8*best_node->idx1[i]+best_node->atb]<best_node->th){
                lchild.idx1[cnt_left++] = best_node->idx1[i];
                code_pf1[best_node->idx1[i]] = nid_left;
            }
            else{
                rchild.idx1[cnt_right++] = best_node->idx1[i];
                code_pf1[best_node->idx1[i]] = nid_right;
            }
        }
        
        if(cnt_left!=best_node->left_N1){
        	mexPrintf("cnt_left = %ld, left_N1 = %ld, cnt_right = %ld, right_N1 = %ld, N1 = %ld\n",cnt_left,best_node->left_N1,cnt_right,best_node->right_N1,best_node->N1);
        	mexErrMsgTxt("left_N1.");
        }
        if(cnt_right!=best_node->right_N1){
        	mexPrintf("cnt_right = %ld, right_N1 = %ld\n",cnt_right,best_node->right_N1);
        	mexErrMsgTxt("right_N1.");
        }
        
        cnt_left = 0, cnt_right = 0;
        for(__int64 i = 0;i<best_node->N2;i++){
            if(pf2[8*best_node->idx2[i]+best_node->atb]<best_node->th){
                lchild.idx2[cnt_left++] = best_node->idx2[i];
                code_pf2[best_node->idx2[i]] = nid_left;
            }
            else{
                rchild.idx2[cnt_right++] = best_node->idx2[i];
                code_pf2[best_node->idx2[i]] = nid_right;
            }
        }
        if(cnt_left!=best_node->left_N2) mexErrMsgTxt("left_N2.");
        if(cnt_right!=best_node->right_N2) mexErrMsgTxt("right_N2.");
        
        tree.current_nb_nodes += 2;
        mxFree(best_node->idx1); best_node->idx1 = 0;
        mxFree(best_node->idx2); best_node->idx2 = 0;
    }
    int code = 0;
    for(int i = 0; i< tree.total_nb_nodes; i++){
    	if(tree.table[i*tree.nb_elem+CODE] != -1){
    		tree.table[i*tree.nb_elem+CODE] = code++;
    	}
    }
    tree.nb_conflicting_pixel_pairs = 0;
    for(__int64 i = 0;i<num_pf;i++){
    	if(code_pf1[i]<0 || code_pf1[i]>=tree.total_nb_nodes) mexErrMsgTxt("code_pf1.");
    	if(code_pf2[i]<0 || code_pf2[i]>=tree.total_nb_nodes) mexErrMsgTxt("code_pf2.");
    	code_pf1[i] = tree.table[code_pf1[i]*tree.nb_elem+CODE];
    	code_pf2[i] = tree.table[code_pf2[i]*tree.nb_elem+CODE];
    	tree.nb_conflicting_pixel_pairs += code_pf1[i] != code_pf2[i];
    }
    mxFree(tree_nodes);
    mxFree(code_pf1);
    mxFree(code_pf2);
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{ 	//images1, images2, ncode, radii, alpha
	
    int nb_images = mxGetNumberOfElements(prhs[0]); // number of images in cell array.
    vector<uchar*> images1,images2;
    for (int n = 0; n < nb_images; n++){
        images1.push_back((uchar*)mxGetPr(mxGetCell(prhs[0],n)));
        images2.push_back((uchar*)mxGetPr(mxGetCell(prhs[1],n)));
    }
    int ncode = *mxGetPr(prhs[2]); //number of leave nodes.
    if(mxGetNumberOfElements(prhs[2])==2){ //verbose mode.
    	mode = 1;
    }else
    	mode = 0;
    int nradii = mxGetNumberOfElements(prhs[3]);
    double alpha = *mxGetPr(prhs[4]);
    aligned = *mxGetPr(prhs[5]); // 1 for aligned, 0 for not aligned.
    
    vector<int>radii; //sampling radii.
    for(int i=0;i<nradii;i++) radii.push_back(*(mxGetPr(prhs[3])+i));
    int height = mxGetM(mxGetCell(prhs[0],0));
    int width = mxGetN(mxGetCell(prhs[0],0));
    __int64 N = height * width * nb_images; //maximum number of pixel pairs.
    int nnodes = 2*ncode-1; //number of leave nodes in the encoding tree.
    short* pf1 = (short*)mxMalloc(N*8*sizeof(short));
    short* pf2 = (short*)mxMalloc(N*8*sizeof(short));
    mxArray *trees = mxCreateCellMatrix(1,nradii);

    const char *fieldnames[] = {"tree","ncode","alpha","clft","cost","prob","radius"};
    plhs[0] = mxCreateStructMatrix(1,nradii,7,fieldnames);
    
    if(aligned){
    	if(mode==1){
			mexPrintf("\nFace alignment...");
			mexEvalString("drawnow;");
    	}
    	align_faces(images1,images2,height,width);
    	if(mode==1){
			mexPrintf("done\n");
			mexEvalString("drawnow;");
    	}
    	/*for(int i = 0;i<height;i++){
    		for(int j = 0;j<width;j++){
    			mexPrintf("%d ",images1[0][i*width+j]);
    		}
    		mexPrintf("\n");
    	}
    	return;*/
    }
    
    for(int i=0;i<nradii;i++){
    	__int64 nb_pf, nb_pf_1, nb_pf_2;
		nb_pf_1 = extract_pixel_feat(pf1, images1, height, width, radii[i]);
        nb_pf_2 = extract_pixel_feat(pf2, images2, height, width, radii[i]);
        nb_pf = nb_pf_1;
        if(nb_pf==0)
        	mexErrMsgTxt("Cannot detect any pair of face.");
        if(nb_pf_1!=nb_pf_2)
        	mexErrMsgTxt("nb_pf_1 != nb_pf_2.");
        ENC_TREE* tree = new ENC_TREE(ncode,nb_pf,alpha);
        train_encode_tree(*tree,pf1,pf2,ncode);
        
        mxArray* p1 = mxCreateDoubleMatrix(tree->nb_elem, nnodes, mxREAL); //table of the encoding tree.
        mxArray* p2 = mxCreateDoubleMatrix(1, 1, mxREAL); //ncode
        mxArray* p3 = mxCreateDoubleMatrix(1, 1, mxREAL); //alpha
        mxArray* p4 = mxCreateDoubleMatrix(1, 1, mxREAL); //conflicting percent.
        mxArray* p5 = mxCreateDoubleMatrix(1, 1, mxREAL); //cost
        mxArray* p6 = mxCreateDoubleMatrix(1, ncode, mxREAL); //probability distribution.
        mxArray* p7 = mxCreateDoubleMatrix(1, 1, mxREAL); //sampling radius.
        
        memcpy(mxGetPr(p1),tree->table,sizeof(double)*tree->nb_elem*nnodes);
        *mxGetPr(p2) = tree->nb_codes;
        *mxGetPr(p3) = tree->alpha;
        *mxGetPr(p4) = (double)tree->nb_conflicting_pixel_pairs/tree->nb_pixel_pairs;
        *mxGetPr(p7) = radii[i];
        
        double cost = tree->alpha*(*mxGetPr(p4));
        double * ptr = tree->table;
        int cnt = 0;
        for(int j = 0;j<tree->total_nb_nodes;j++){
            if(ptr[CODE]>=0){
                double p = ptr[NSMPL]/(nb_images*2.0*height*width);
                *(cnt+mxGetPr(p6)) = p;
                if(tree->nb_codes>1)
                    cost += (1-tree->alpha)*p*log(p)/log(tree->nb_codes);
                cnt++;
                if(cnt>tree->nb_codes) 
                    mexErrMsgTxt("The number of leaf nodes is inconsistent.");
            }
            ptr += tree->nb_elem;
        }
        *mxGetPr(p5) = cost;
        
        mxSetFieldByNumber(plhs[0],i,0, p1);
        mxSetFieldByNumber(plhs[0],i,1, p2);
        mxSetFieldByNumber(plhs[0],i,2, p3);
        mxSetFieldByNumber(plhs[0],i,3, p4);
        mxSetFieldByNumber(plhs[0],i,4, p5);
        mxSetFieldByNumber(plhs[0],i,5, p6);
        mxSetFieldByNumber(plhs[0],i,6, p7);
		delete tree;
    }
    mxFree(pf1);
    mxFree(pf2);
    if(aligned){
    	for (int n = 0; n < nb_images; n++)
    		if(images1[n] && images2[n])
    			delete images1[n],images2[n];
    }
}

