#include "neuropl.cpp"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

void printRawResult(auto& result){
    for (auto output : result) {
            for (auto item : output) {
                printf("%d ", item);
            }
            printf("\n");
        }
}

int main(void){
    // edit model & test img path here:
    string model_path = "./../openpose_mobilenetv0.75_quant_1x368x368x3.dla";
    string test_img_path = "./../yuzuru.jpg";

    // initialize objects
    Neuropl model {model_path};
    Mat img = imread(test_img_path);
    vector<uint8_t> outputSize = {1,46,46,57};

    // crop and resize img
    auto w = img.size[1], h = img.size[0]; // type int
    auto target_len = min(w, h);
    auto start_x = (w-target_len)/2, start_y = (h-target_len)/2;
    Mat cropped = img(Range(start_y, start_y+target_len), Range(start_x, start_x+target_len));
    Mat resized;
    resize(cropped, resized, Size(368, 368));
    Mat img_RGB;
    cvtColor(resized, img_RGB, COLOR_BGR2RGB);

    // predict
    uint8_t* byte_buffer = reinterpret_cast<uint8_t*>(img_RGB.data);
    auto result = model.predict(byte_buffer);

    // print raw result
    printRawResult(result);

    // process result
    vector output = result[0]; // extract single output

    // render result image

    return 0;
}