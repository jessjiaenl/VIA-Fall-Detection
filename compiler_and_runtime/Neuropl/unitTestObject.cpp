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

void renderBox(string windowName, Mat image, Point topLeft, Point botRight){
    // Create a black screen
    Mat screen(Size(800,800), CV_8UC3, Scalar(0, 0, 0));
    // Calculate the position to place the image at the center
    int x = (screen.cols - image.cols) / 2;
    int y = (screen.rows - image.rows) / 2;
    
    //arguments: image: which image to draw on, top left of the box, bottom right of the box, colour, thickness
    rectangle(image, topLeft, botRight, Scalar(0, 0, 255), 2);
    // Copy the image onto the black screen at the center
    image.copyTo(screen(Rect(x, y, image.cols, image.rows)));
    //Display the image in the window:
    imshow(windowName, screen);
    //Wait for a keyboard event and handle it:
    waitKey(0);
    //Destroy the window and release resources:
    destroyWindow(windowName);
}

int main(void){
    // edit model & test img path here:
    string model_path = "./../mobilenet_ssd_pascal_quant.dla";
    string test_img_path = "./../yuzuru.jpg";

    // initialize objects
    Neuropl model {model_path};
    Mat img = imread(test_img_path);

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
    vector output = result[0]; // extract single output
    cout << typeof(output).name() << endl;

    // process result


    // render result image
    
    return 0;
}