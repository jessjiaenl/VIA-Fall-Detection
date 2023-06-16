#include <chrono>
#include <iostream>
#include <fstream>
#include "TFLib.h"


bool part_run;


void pause(){
	if (part_run) {
		printf("Press any key to continue...\n");
		std::cin.get();
	}
}


void print_tensor(TFLiteTensor *tensor,int topN){
	uint8_t *result_q=reinterpret_cast<uint8_t*>(tensor->buffer);
	float *result_f=reinterpret_cast<float*>(tensor->buffer);
	bool IsQuant=(tensor->type==2);

	if (topN<1) {
		topN=tensor->bufferSize;
	}
	for (size_t i=0 ; i< topN ; i++){
		if (i==0){
			printf("[");
		}
		if (IsQuant) {printf("\t%d",result_q[i]); }else{ printf("\t%5.5f",result_f[i]); }
		if ((i!=0)&&((i+1)%5==0)&&(i<topN-1)){
			printf("\t]\n[");
		}
		//printf("result[%zu]: %f\n",i ,result[i]);
	}
	printf("\t]\n\n");
}

ANeuralNetworksTFLite* model_init(const char* model_path,const char* input_path){
	ANeuralNetworksTFLite* tf;
	auto time1 = std::chrono::high_resolution_clock::now();
	if (ANeuralNetworksTFLite_create(&tf, model_path) != ANEURALNETWORKS_NO_ERROR){
		return 0;
	} 
	auto time2 = std::chrono::high_resolution_clock::now();
	printf("Create and init time: %f ms\n",float((time2-time1).count())/1000000);
	
	TFLiteTensor inputTensor;
	if (ANeuralNetworksTFLite_getTensor(tf, 
										TFLITE_BUFFER_TYPE_INPUT, 
										&inputTensor) != ANEURALNETWORKS_NO_ERROR){
		ANeuralNetworksTFLite_free(tf);
		return 0;
	}

	// Fill input data from file
	std::ifstream input(input_path);
	if (!input.good()) {
		printf("Fail to read %s\n", input_path);
		return 0;
	}

	if (inputTensor.type == 2) {  		// Is uint8
		input.read((char*)inputTensor.buffer, sizeof(uint8_t) * inputTensor.bufferSize);
	}else{								// Is float
		input.read((char*)inputTensor.buffer, sizeof(float) * inputTensor.bufferSize);
	}
	//print_tensor(&inputTensor,20);
	input.close();

	/*
	float *ptr_box = (float *)inputTensor.buffer;
	for(int i=0;i<20;i++){ //inputTensor.bufferSize
		printf(",%f",*ptr_box);
		ptr_box++;
	}
	*/


	auto time3 = std::chrono::high_resolution_clock::now();
	printf("Get input tensor , time: %f ms , type: %d , dimsSize: %d , bufferSize: %zu , dims=[ ",
		float((time3-time2).count())/1000000,inputTensor.type,inputTensor.dimsSize,inputTensor.bufferSize);
	for (int i=0 ; i<inputTensor.dimsSize ; i++)
		printf("%d ", inputTensor.dims[i]);
	printf("]\n");
		
	return tf;
}

int model_inference(ANeuralNetworksTFLite* tf,bool showtime){
	auto ti1 = std::chrono::high_resolution_clock::now();
	if (ANeuralNetworksTFLite_invoke(tf) != ANEURALNETWORKS_NO_ERROR){
		ANeuralNetworksTFLite_free(tf);
		return 0;
	}
	auto ti2 = std::chrono::high_resolution_clock::now();
	if (showtime){
		printf("Inference time: %f ms\n",float((ti2-ti1).count())/1000000);
	}
}


int model_output(ANeuralNetworksTFLite* tf,int index,bool show_tensor){
	auto time6 = std::chrono::high_resolution_clock::now();
	TFLiteTensor outputTensor;
	if (ANeuralNetworksTFLite_getTensorByIndex(tf, 
										TFLITE_BUFFER_TYPE_OUTPUT, 
										&outputTensor,
										index) != ANEURALNETWORKS_NO_ERROR){
		ANeuralNetworksTFLite_free(tf);
		return 0;
	}
	auto time7 = std::chrono::high_resolution_clock::now();
	printf("Get output tensor , time: %f ms , type: %d , dimsSize: %d , bufferSize: %zu , dims=[ ",
		float((time7-time6).count())/1000000,outputTensor.type,outputTensor.dimsSize,outputTensor.bufferSize);
	for (int i=0 ; i<outputTensor.dimsSize ; i++)
		printf("%d ", outputTensor.dims[i]);
	printf("]\n");

	if (show_tensor){
		print_tensor(&outputTensor,20);
	}	

}


