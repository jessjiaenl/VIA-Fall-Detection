#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <chrono>

#include "TFLib.h"
#include "ssd.h"

using namespace std;
ANeuralNetworksTFLite *tf1;


typedef struct {
    const char *path;
    const char *input;
} model_type;


void model_run(model_type mt){
	char ResultMSG[2000];

	printf("\n-------- Load model --------\n\n");
	tf1=model_init(mt.path,mt.input);

	printf("\n-------- Inference --------\n\n");
	auto t1 = std::chrono::high_resolution_clock::now();
	model_inference(tf1,true);
	model_output(tf1,0,false);
	model_output(tf1,1,false);
	ssd_post_process(tf1,true,ResultMSG);	
	auto t2 = std::chrono::high_resolution_clock::now();
	float t=float((t2-t1).count())/1000000;
	printf("\nTotal time: %0.3f ms\n", t);	

	ANeuralNetworksTFLite_free(tf1);
	printf("\n----- UnitTest verify -----\n\n");
	if (t<80){
		printf("Performance correct\n");
	}

	if ((strstr(ResultMSG,"person")>0)&&(strstr(ResultMSG,"motorbike")>0)){
		printf("Accuracy correct\n");
	}
}

int main(int argc, char** argv)
{
	(void)(argc);
	(void)(argv);

	const model_type mt={ "mobilenet_ssd_pascal_quant.tflite","mobilenet_ssd_input_q.bin" };		//APU

	model_run(mt);

	return 1;
}

