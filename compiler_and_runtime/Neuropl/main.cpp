#include "neuropl.cpp"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

int predictNRenderVid(int modelidx, Neuropl& model, string vid_path){
    VideoCapture cap(vid_path);
    Mat frame; // is of type 8UC3 (3 channels)
    bool readSucc = cap.read(frame);
    
    // crop & resize frame
    auto w = frame.size[1], h = frame.size[0]; // type int
    auto target_len = min(w, h);
    auto start_x = (w-target_len)/2, start_y = (h-target_len)/2;
    Mat cropped_frame = frame(Range(start_y, start_y+target_len), Range(start_x, start_x+target_len));
    Mat resized_frame;
    auto size = Size(224, 224);
    if (modelidx == 0) size = Size(368, 368);
    else if (modelidx == 1) size = Size(300, 300);
    resize(cropped_frame, resized_frame, size);
    Mat frame_RGB;
    cvtColor(resized_frame, frame_RGB, COLOR_BGR2RGB);

    // start predicting frame by frame
    while(true){
        if (!readSucc){cap.set(CAP_PROP_POS_FRAMES, 0);}
        uint8_t* byte_buffer = reinterpret_cast<uint8_t*>(frame.data);
        auto result = model.predict(byte_buffer);

        for (auto output : result) {
            for (auto item : output) {
                printf("%d ", item);
            }
            printf("\n");
        }
        break;
    }
    return 0;
}

int main(void){
    vector<string> model_paths;
    //model_paths.push_back("");
    model_paths.push_back("./../openpose_mobilenetv0.75_quant_1x368x368x3.dla");
    model_paths.push_back("./../mobilenet_ssd_pascal_quant.dla");
    
    int modelidx = 1;
    bool useVid = true;
    string vid_path = "./../jess_IMG_0480.MOV";

    string model_path = model_paths[modelidx];
    Neuropl model {model_path};

    predictNRenderVid(modelidx, model, vid_path);

    return 0;
}