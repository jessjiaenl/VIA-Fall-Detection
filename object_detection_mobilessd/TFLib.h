#include <android/NeuralNetworks.h>
#ifdef ANDROID_P
	#include "NeuroPilotTFLiteShim.h"
#elif ANDROID_O
	#include <mtk/NeuralNetworksMTK.h>
#endif

extern void pause();
extern void print_tensor(TFLiteTensor *tensor,int topN);
extern int model_inference(ANeuralNetworksTFLite* tf,bool showtime);
extern int model_output(ANeuralNetworksTFLite* tf,int index,bool show_tensor);

extern ANeuralNetworksTFLite* model_init(const char* model_path,const char* input_path);