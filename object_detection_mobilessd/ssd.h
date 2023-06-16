#include "pascal.h"


#define TOP_N_RESULT                   (2)
#define TOP_N_INCLASS                  (2)
#define RESULT_NUM                     (1917)
#define IMG_DIMENSION                  (300)
#define BBOX_DIMENSION                 (4)
#define IOU_THRESHOLD                  (0.6)
#define MIN_CONFIDENCE                 (0.5)

#define X_SCALE                        (10.0f)
#define Y_SCALE                        (10.0f)
#define W_SCALE                        (5.0f)
#define H_SCALE                        (5.0f)

#define WIN_XMIN					   (0)
#define WIN_XMAX 					   (300)
#define WIN_YMIN 					   (0)
#define WIN_YMAX 					   (300)


extern int ssd_post_process(ANeuralNetworksTFLite *tflite,bool showlog,char *return_msg);

