#include <prmi/parallel.h>

void* Parallel:: execute_thread(void* arg){
    ARG_EXE_THREAD* a = (ARG_EXE_THREAD*)arg;
    a->func(a->param);
}

void Parallel::Run(void (*f)(void*),std::vector<void*>& param){
    nw = param.size();
	if(tid) delete tid;
    tid = new pthread_t [nw];
    //startup threads.
	if(args) delete args;
    args = new ARG_EXE_THREAD [nw];
    for(int i=0;i<nw;i++)
    {
        args[i].param = param[i];
        args[i].func = f;
        pthread_create(&(tid[i]), NULL, &execute_thread, &args[i]);
    }
	//wait until finish.
    for(int i=0;i<nw;i++)
        pthread_join(tid[i], NULL);
}

Parallel::~Parallel(){
    delete args;
    delete tid;
	tid = 0;
	args = 0;
}
Parallel::Parallel(){
	args = 0; 
	tid = 0;
}
