#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <android/NeuralNetworks.h>
#include "TFLib.h"


// the label &box prior information
#include "ssd.h"

// data structure for storing the inference result
typedef struct
{
	float x;
	float y;
	float w;
	float h;
} bbox_t;

typedef struct
{
	bbox_t bbox;
	int class_id;
	float confidence;
	float *class_score;
	bool invalid;
} info_t;

typedef struct
{
	int box_id;
	int class_id;
	float confidence;
	bool invalid;
} classbox_t;

const int ALL_RESULT_SIZE = TOP_N_INCLASS*CLASS_NUM;

// compare function for qsort
int _compare(void const *a, void const *b);
int _classbox_compare(void const *a, void const *b);
int _resultbox_compare(void const *a, void const *b);

// print tensor information
void _print_tensor_info(const char *name, TFLiteTensor *tensor);
// expit (Sigmoid) function
float _expit(float x);
// retrive the class with max confidence
void _get_max_confidence(float *input, int *class_index, float *confidence);
// retrive the bounding box information
void _get_decoded_box(float *input, int region_id, bbox_t *bbox);
// retrive the label
const char *_get_label(int class_id);
// calculate the IoU of two rectangles
float _get_iou(bbox_t *box_a, bbox_t *box_b);
// clip bounding box to certain window
int _clip_to_window(info_t *data, bbox_t *window);
// non max supression calculation
int _perform_non_max_supression(info_t *data, int current_id);
int _perform_non_max_supression_byclass(classbox_t *data, info_t *boxlist, int classid, int current_id);


float *dequant_to_float_buffer(TFLiteTensor *optTensor,float mean,float step){
	size_t len=optTensor->bufferSize;
	float *result_ptr = (float*) calloc (len,sizeof(float));
	uint8_t *uptr=reinterpret_cast<uint8_t*>(optTensor->buffer);

	if (optTensor->type != 2) {
		result_ptr = (float *)optTensor->buffer;
	}else{
		float *data=result_ptr;
		for(size_t i=0;i<len;i++){
			*data= ((float)(*uptr) - mean )*step;
			uptr++;
			data++;
		}
	}
	return result_ptr;
}


int ssd_post_process(ANeuralNetworksTFLite *tflite,bool showlog,char *return_msg){
	
	TFLiteTensor outputTensor_box, outputTensor_class;
	int i, j;
	float *box_ptr, *class_ptr;
	
	bbox_t clipwindow;
	clipwindow.x = WIN_XMIN;
	clipwindow.y = WIN_YMIN;
	clipwindow.w = WIN_XMAX - WIN_XMIN;
	clipwindow.h = WIN_YMAX - WIN_YMIN;
	
	
	auto t5 = std::chrono::high_resolution_clock::now();	
	if (ANeuralNetworksTFLite_getTensorByIndex(				//box: concat - 1 x 1917 x 4 
		tflite, 
		TFLITE_BUFFER_TYPE_OUTPUT, 
		&outputTensor_box,
		1) != ANEURALNETWORKS_NO_ERROR)
	{
		if (showlog) printf("Failed to the information of the output tensor.\n");
		ANeuralNetworksTFLite_free(tflite);
		return -1;
	}

	if (ANeuralNetworksTFLite_getTensorByIndex(				//class: concat_1 - 1 x 1917 x 91  
		tflite,
		TFLITE_BUFFER_TYPE_OUTPUT,
		&outputTensor_class,
		0) != ANEURALNETWORKS_NO_ERROR)
	{
		if (showlog) printf("Failed to the information of the output tensor.\n");
		ANeuralNetworksTFLite_free(tflite);
		return -1;
	}

	// show the information about the output tensor
	//if (showlog) _print_tensor_info("Output (box)", &outputTensor_box);
	//if (showlog) _print_tensor_info("Output (class)", &outputTensor_class);

	// allocate memory for storing the data to be sorted
	info_t *data = (info_t *)calloc(RESULT_NUM, sizeof(info_t));
	if (data == NULL)
	{
		if (showlog) printf("Failed to allocate memory.\n");
		ANeuralNetworksTFLite_free(tflite);
		return -1;
	}	

	classbox_t *classboxlist[CLASS_NUM];
	for (i=0; i<CLASS_NUM; i++) 
	{
		classboxlist[i] = (classbox_t *)calloc(RESULT_NUM, sizeof(classbox_t));
		if (classboxlist[i] == NULL)
		{
			if (showlog) printf("Failed to allocate memory.\n");
			ANeuralNetworksTFLite_free(tflite);
			return -1;
		}	
	}
	
	classbox_t **classresultlist;
	classresultlist = (classbox_t **)calloc(ALL_RESULT_SIZE, sizeof(classbox_t *));
	if (classresultlist == NULL)
	{
		if (showlog) printf("Failed to allocate memory.\n");
		ANeuralNetworksTFLite_free(tflite);
		return -1;
	}	


	
	box_ptr   = dequant_to_float_buffer(&outputTensor_box,DEQUANT_BBOX_MEAN,DEQUANT_BBOX_STEP);
	class_ptr = dequant_to_float_buffer(&outputTensor_class,DEQUANT_CLASS_MEAN,DEQUANT_CLASS_STEP);
	// box_ptr   = (float *)outputTensor_box.buffer;
	// class_ptr = (float *)outputTensor_class.buffer;		

	// fill in the result
	for (i = 0; i < RESULT_NUM; i++)
	{
		// get bounding box information
		_get_decoded_box(box_ptr, i, &data[i].bbox);
		data[i].class_score = class_ptr;
		// get max confidence of the class
		_get_max_confidence(class_ptr, &data[i].class_id, &data[i].confidence);
		// get bounding box information
		for (j = 1; j < CLASS_NUM; j++)  
		{
			classboxlist[j][i].box_id = i;
			classboxlist[j][i].class_id = j;
			classboxlist[j][i].confidence = _expit(class_ptr[j]);
		}
		_clip_to_window(&data[i], &clipwindow);
		
		// advance to the next result
		box_ptr   += BBOX_DIMENSION;
		class_ptr += CLASS_NUM;				
	}	

	//perforrm per class NMS
	for (j = 1; j < CLASS_NUM; j++) 
	{
		// sort the result (descending)
		int current_item = 0, next_item;
		qsort(classboxlist[j], RESULT_NUM, sizeof(classbox_t), _classbox_compare);
		if (classboxlist[j][0].invalid)
			continue;
		if (classboxlist[j][0].confidence < MIN_CONFIDENCE)
		{
			classboxlist[j][0].invalid = true;
			continue;
		}		
		for (i = 0; (i < TOP_N_INCLASS) && (current_item < RESULT_NUM); i++)
		{
			// perform non max supression and get the next item to be output
			next_item = _perform_non_max_supression_byclass(classboxlist[j], data, j, current_item);
			classresultlist[j*TOP_N_INCLASS+i] = &(classboxlist[j][current_item]);
			//printf("class:%d, top:%d, conf:%f\n", j, i, classboxlist[j][0].confidence  );

			// no more valid output
			if (next_item < 0)
			{
				break;
			}
			current_item = next_item;
		}
	}
 
  //squeeze classresultlist, remove NULL items
  int rescnt = 0;
  for (i=0; i<ALL_RESULT_SIZE; i++) {
    if (classresultlist[i]!=NULL) {
      classresultlist[rescnt] = classresultlist[i];
      rescnt++;
    }
  }
  if (showlog) printf("\nTotal %d detections found:\n", rescnt);
	// show top N result
	qsort(classresultlist, rescnt, sizeof(classbox_t *), _resultbox_compare);
  int outputsize = (rescnt>TOP_N_RESULT)?TOP_N_RESULT:rescnt;
	if (showlog) printf("\nTop %d Results:\n", outputsize);
	
	for (i=0; i<outputsize; i++) 
	{
		classbox_t *pclassbox = classresultlist[i];
		if (!pclassbox)
			continue;
		int current_item = pclassbox->box_id;
		if ((!pclassbox->invalid) && (!data[current_item].invalid))
		{
			
			if ((current_item<0) || (current_item>RESULT_NUM))
				printf("Something wrong\n");
			// output msg
			if (return_msg!=NULL){
				sprintf(return_msg, "%s%s\t%d\t%f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\n",
					return_msg,
					_get_label(pclassbox->class_id),
					pclassbox->class_id,
					pclassbox->confidence,
					data[current_item].bbox.x,
					data[current_item].bbox.y,
					data[current_item].bbox.x+data[current_item].bbox.w,
					data[current_item].bbox.y+data[current_item].bbox.h);
			}


			// show top N result	
			if (showlog) printf(
				"[Top %d\t %0.4f] %s \tX:%3.2f \tY:%3.2f \tW:%3.2f \tH:%3.2f \tBoxID:%d\n", 
				i + 1, 
				pclassbox->confidence, 
				_get_label(pclassbox->class_id),
				data[current_item].bbox.x,
				data[current_item].bbox.y,
				data[current_item].bbox.w,
				data[current_item].bbox.h,
				pclassbox->box_id);
		}
	}
	auto t6 = std::chrono::high_resolution_clock::now();
	printf("\nPost-process time: %f ms",float((t6-t5).count())/1000000);

	// release allocated memory
	free(data);
}


// compare function for qsort
int _compare(void const *a, void const *b)
{
	info_t *x, *y;

	x = (info_t *)a;
	y = (info_t *)b;

	return (x->confidence < y->confidence)? 1: -1;
}


// compare function for qsort
int _classbox_compare(void const *a, void const *b)
{
	classbox_t *x, *y;

	x = (classbox_t *)a;
	y = (classbox_t *)b;

	if (x->invalid) {
		if (y->invalid)
			return 0;
		else
			return 1;
	}
	if (y->invalid) {
		return -1;
	}
	return (x->confidence < y->confidence)? 1: -1;
}

// compare function for qsort
int _resultbox_compare(void const *a, void const *b)
{
	classbox_t **x, **y;

	x = (classbox_t **)a;
	y = (classbox_t **)b;
	if (*x==NULL) {
		if (*y = NULL)
			return 0;
		else
			return 1;
	}
	if (*y==NULL) {
		return -1;
	}
	return ((*x)->confidence < (*y)->confidence)? 1: -1;
}


// print tensor information
void _print_tensor_info(const char *name, TFLiteTensor *tensor)
{
	printf("\n%s tensor\n", name);
	printf("type : %d\n", tensor->type);
	printf("dimsSize : %d\n", tensor->dimsSize);

	for (int i = 0; i < tensor->dimsSize; i++)
	{
		printf("%d ", tensor->dims[i]);
	}

	printf("\nbufferSize : %zu\n", tensor->bufferSize);
}

// expit (Sigmoid) function
float _expit(float x)
{
	return (1.0 / (1.0 + exp(-x)));
}

// retrive the class with max confidence
void _get_max_confidence(float *input, int *class_index, float *confidence)
{
	int i;
	float max = -1.0, adjusted;

	// Skip the first catch-all class
	input++;

	for (i = 1; i < CLASS_NUM; i++)
	{
		float data= *input;
		adjusted = _expit(data);

		if (adjusted > max)
		{
			max = adjusted;

			*class_index = i;
		}

		input++;
	}

	*confidence = max;
}

// retrive the bounding box information
void _get_decoded_box(float *input, int region_id, bbox_t *bbox)
{
	float x_center, y_center;
	float w, h;

	float prior_x, prior_y, prior_w, prior_h;

	prior_x = _box_prior[1 * RESULT_NUM + region_id];
	prior_y = _box_prior[0 * RESULT_NUM + region_id];

	prior_w = _box_prior[3 * RESULT_NUM + region_id];
	prior_h = _box_prior[2 * RESULT_NUM + region_id];
	

	y_center = input[0] / Y_SCALE * prior_h + prior_y;
	x_center = input[1] / X_SCALE * prior_w + prior_x;

	h = exp(input[2] / H_SCALE) * prior_h;
	w = exp(input[3] / W_SCALE) * prior_w;


	bbox->x = IMG_DIMENSION * (x_center - w / 2);
	bbox->y = IMG_DIMENSION * (y_center - h / 2);
	bbox->w = IMG_DIMENSION * w;
	bbox->h = IMG_DIMENSION * h;
}

// retrive the label
const char *_get_label(int class_id)
{
	if (class_id >= CLASS_NUM)
	{
		return NULL;
	}

	return _label[class_id];
}

#define MIN(x, y)               (((x) > (y))? (y): (x))
#define MAX(x, y)               (((x) < (y))? (y): (x))
#define IS_VALID_RECT(bbox)     ((bbox).w > 0 && (bbox).h > 0)
#define RECT_AREA(bbox)         ((bbox).w * (bbox).h)

// calculate the IoU of two rectangles
float _get_iou(bbox_t *box_a, bbox_t *box_b)
{
	bbox_t overlap;

	overlap.x = MAX(box_a->x, box_b->x);
	overlap.y = MAX(box_a->y, box_b->y);

	overlap.w = MIN(box_a->x + box_a->w, box_b->x + box_b->w) - overlap.x;
	overlap.h = MIN(box_a->y + box_a->h, box_b->y + box_b->h) - overlap.y;

	if (!IS_VALID_RECT(overlap))
	{
		return 0.0f;
	}

	return RECT_AREA(overlap) / (RECT_AREA(*box_a) + RECT_AREA(*box_b) - RECT_AREA(overlap));
}


int _clip_to_window(info_t *data, bbox_t *window) 
{
	if ((data==NULL)||(window==NULL))
		return -1;
	float wx1 = window->x;
	float wx2 = wx1 + window->w;
	float wy1 = window->y;
	float wy2 = wy1 + window->h;	
	float bx1 = data->bbox.x;
	float by1 = data->bbox.y;	
	float bx2 = data->bbox.x + data->bbox.w;
	float by2 = data->bbox.y + data->bbox.h;	
	if (bx1 < wx1) bx1 = wx1;
	if (bx2 < wx1) bx2 = wx1;
	if (bx1 > wx2) bx1 = wx2;
	if (bx2 > wx2) bx2 = wx2;
	if (by1 < wy1) by1 = wy1;
	if (by2 < wy1) by2 = wy1;
	if (by1 > wy2) by1 = wy2;
	if (by2 > wy2) by2 = wy2;	
	data->bbox.x = bx1;
	data->bbox.y = by1;
	data->bbox.w = bx2-bx1;
	data->bbox.h = by2-by1;
	if (data->bbox.w*data->bbox.h < 1e-5) 		
	{
		data->invalid = true;
	}		
	
	return 0;
}



// non max supression calculation
int _perform_non_max_supression(info_t *data, int current_id)
{
	int i, next_item = -1;

	for (i = current_id + 1; i < RESULT_NUM; i++)
	{
		if (data[i].invalid)
		{
			continue;
		}

		if (data[i].confidence < MIN_CONFIDENCE)
		{
			data[i].invalid = true;
			continue;
		}

		if (data[i].class_id == data[current_id].class_id && _get_iou(&data[i].bbox, &data[current_id].bbox) > IOU_THRESHOLD)
		{
			data[i].invalid = true;
			continue;
		}

		// first candidate for the next item
		if (next_item < 0)
		{
			next_item = i;
		}
	}

	return next_item;
}


// non max supression calculation
int _perform_non_max_supression_byclass(classbox_t *classboxlist, info_t *boxlist, int classid, int current_id)
{
	int i, next_item = -1;
	
	if (classboxlist[current_id].class_id != classid)
		return -1;
	
	int curid = current_id;
	
	for (i = curid + 1; i < RESULT_NUM; i++)
	{
		if (classboxlist[i].invalid)
		{
			continue;
		}

		if (classboxlist[i].confidence < MIN_CONFIDENCE)
		{
			classboxlist[i].invalid = true;
			continue;
		}

		int boxid = classboxlist[i].box_id;
		int curboxid = classboxlist[curid].box_id;
		if (boxlist[boxid].invalid) {
			classboxlist[i].invalid = true;
			continue;
		}	
		
		if (classboxlist[i].class_id == classid && _get_iou(&boxlist[boxid].bbox, &boxlist[curboxid].bbox) > IOU_THRESHOLD)
		{
			classboxlist[i].invalid = true;
			continue;
		}

		if (next_item < 0)
		{
			next_item = i;
		}
	}

	return next_item;
}
