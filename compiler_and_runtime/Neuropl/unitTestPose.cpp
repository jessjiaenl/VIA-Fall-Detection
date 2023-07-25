#include "neuropl/neuropl.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

const int WIDTH = 368, HEIGHT = 368; // model input img size

/* global variables needed for rendering */
const int SCREEN_SIZE = 500;
const double STRIKE_WIDTH_TEXT = 8.0;
const double STRIKE_WIDTH_LINE = 3.0;
const int TOTAL_POINTS = 18;
// BGR
const  Scalar COLOR_HEAD(66, 206, 245);
const  Scalar COLOR_BODY(45, 224, 117);
const  Scalar COLOR_FOOT(224, 45, 140);
const  Scalar COLOR_LINE(71, 78, 255);

const vector<Scalar> POINT_COLORS = {
    COLOR_HEAD,   // 0. nose
    COLOR_BODY,   // 1. neck
    COLOR_BODY,   // 2. R shoulder
    COLOR_BODY,   // 3. R elbow
    COLOR_BODY,   // 4. R wrist
    COLOR_BODY,   // 5. L shoulder
    COLOR_BODY,   // 6. R elbow
    COLOR_BODY,   // 7. R wrist
    COLOR_FOOT,   // 8. R hip
    COLOR_FOOT,   // 9. R knee
    COLOR_FOOT,   // 10. R ankle
    COLOR_FOOT,   // 11. L hip
    COLOR_FOOT,   // 12. L knee
    COLOR_FOOT,   // 13. L ankle
    COLOR_HEAD,   // 14. R eye
    COLOR_HEAD,   // 15. L eye
    COLOR_HEAD,   // 16. R ear
    COLOR_HEAD,   // 17. L ear
};

const vector<tuple<int, int>> LINE_POINTS = {
    make_tuple(0, 1),
    make_tuple(1, 2),
    make_tuple(2, 3),
    make_tuple(3, 4),
    make_tuple(1, 5),
    make_tuple(5, 6),
    make_tuple(6, 7),
    make_tuple(1, 8),
    make_tuple(8, 9),
    make_tuple(9, 10),
    make_tuple(1, 11),
    make_tuple(11, 12),
    make_tuple(12, 13),
    make_tuple(0, 14),
    make_tuple(0, 15),
    make_tuple(14, 16),
    make_tuple(15, 17)
};

vector<string> faceIndex = {"0", "14", "15", "16", "17"};
vector<tuple<double,double>> points;
Mat canvas;
/* end of global vars */

/* initialize canvas */
void initCanvas(Mat image){
    canvas = image;
    points.resize(TOTAL_POINTS);
}

/* record latest data, which will be shown in the next call of drawPose(). */
void track(vector<tuple<double,double>> nodePosition, int w, int h) {
    for (int i = 0; i < TOTAL_POINTS; ++i) {
        if (i < nodePosition.size() && get<0>(nodePosition[i]) >= 0) {
            // Resize output based on size of the view.
            points[i] = make_tuple(get<0>(nodePosition[i])/ 46 * WIDTH, get<1>(nodePosition[i]) / 46 * HEIGHT);
        } else {
            // use -1 as the placeholder for NULL
            points[i] = make_tuple(-1, -1);
        }
    }
}

/* draw points and lines */
void drawPose(const vector<tuple<double, double>>& points, Mat& canvas, Mat& img, string windowName) {
    /* draw points */
    for (int i = 0; i < TOTAL_POINTS; ++i) {
        tuple<double, double> point = points[i];
        if (get<0>(point) != -1){
            bool isFacePoint = find(faceIndex.begin(), faceIndex.end(), to_string(i)) != faceIndex.end();
            int x = get<0>(point), y = get<1>(point);
            Point p = Point(x, y); // tuple to point
            printf("x = %d, y = %d\n", x, y);
            circle(canvas, p, (isFacePoint ? 4 : 6), (isFacePoint ? POINT_COLORS[0] : POINT_COLORS[i]), 3); // draw point
        }
    }

    /* draw lines */
    for (const auto& linePoint : LINE_POINTS) {
        int startIndex = get<0>(linePoint), endIndex = get<1>(linePoint);
        tuple<int, int> startPoint = points[startIndex], endPoint = points[endIndex];
        if (get<0>(startPoint) != -1 && get<0>(endPoint) != -1) {
            Point start = Point(get<0>(startPoint), get<1>(startPoint)); // tuple to point
            Point end = Point(get<0>(endPoint), get<1>(endPoint)); // tuple to point
            line(canvas, start, end, COLOR_LINE, STRIKE_WIDTH_LINE); // draw line
        }
    }
    
    /* display */
    imshow("Image", canvas); 
    waitKey(0); // wait for a keyboard event
    destroyWindow("Image"); // then destroy the window and release resources
}

/* print raw predict results (vector of vector) */
void printRawResult(auto& result){
    for (auto output : result) {
            for (auto item : output) {
                printf("%d ", item);
            }
            printf("\n");
        }
}

/* crop and resize the image to model's input size */
void cropNResize(Mat& img, Mat& result, int frame_w, int frame_h){
    auto w = img.size[1], h = img.size[0]; // type int
    auto target_len = min(w, h);
    auto start_x = (w-target_len)/2, start_y = (h-target_len)/2;
    Mat cropped = img(Range(start_y, start_y+target_len), Range(start_x, start_x+target_len));
    Mat resized;
    resize(cropped, result, Size(frame_w, frame_h));
}

/* reshape result from 1d vector to 4d vector (1x46x46x57) */
vector<vector<vector<vector<uint8_t>>>> reshapeResult(vector<uint8_t>& output, vector<int>& outputSize){
    vector<vector<vector<vector<uint8_t>>>> result; // 4 dim vector with shape [1,46,46,57]
    uint8_t inner_cnt = 0, mid_cnt = 0, outer_cnt = 0;
    vector<uint8_t> inner; // with len 57
    vector<vector<uint8_t>> mid; // with len 46
    vector<vector<vector<uint8_t>>> outer; // with len 46
    
    for (int i = 0; i < output.size(); i++){
        inner.push_back(output[i]); // no aliases
        inner_cnt ++;
        if (inner_cnt >= outputSize[3]){
            mid.push_back(inner);
            mid_cnt ++;
            inner.clear();
            inner_cnt = 0;
        }
        if (mid_cnt >= outputSize[2]){
            outer.push_back(mid);
            outer_cnt ++;
            mid.clear();
            mid_cnt = 0;
        }
        if (outer_cnt >= outputSize[1]){
            result.push_back(outer);
            outer.clear();
            outer_cnt = 0;
        }
    }
    return result;
}

/* generates coords and classes to be drawn */
vector<tuple<double, double>> processResult(vector<vector<vector<vector<uint8_t>>>>& result, vector<int> outputSize, double THRESHOLD_CONFIDENCE){
    vector<tuple<double, double>> classCoords(18, make_tuple(-1,-1));
    vector<float> classConfidences(18, THRESHOLD_CONFIDENCE);
    for (int i = 0; i < outputSize[1]; i++){
        for (int j = 0; j < outputSize[2]; j++){ // for each of the boxes
            vector<uint8_t>::iterator maxConfPtr = max_element(result[0][i][j].begin(), result[0][i][j].end());
            uint8_t maxClass = distance(result[0][i][j].begin(), maxConfPtr);
            if (maxClass < 18 && (*maxConfPtr) >= classConfidences[maxClass]){
                classCoords[maxClass] = make_tuple(j,i);
                classConfidences[maxClass] = (*maxConfPtr);
            }
        }
    }
    return classCoords;
}

int main(void){
    /* edit model & test parameters here: */
    string model_path = "./../dla_models/openpose_mobilenetv0.75_quant_1x368x368x3.dla";
    string test_img_path = "./../stand.jpeg";
    vector<int> outputSize = {1,46,46,57}; // model input img size (w,h) is defined as a macro above
    
    double OUTPUT_RATIO = 0.008926701731979847;
    double OUTPUT_BIAS = 126;
    double THRESHOLD_CONFIDENCE = 0.2/OUTPUT_RATIO + OUTPUT_BIAS;

    /* initialize objects */
    auto model = neuropl::Model(model_path);
    namedWindow("Image");
    Mat img = imread(test_img_path);
    Mat img_RGB;
    cropNResize(img, img_RGB, WIDTH, HEIGHT);
    img_RGB.convertTo(img_RGB, CV_8UC3);
    initCanvas(img_RGB); // init render

    /* predict */
    auto pred_result = model.predict(img_RGB.ptr());

    /* print raw result */
    // printf("output count: %lu, output size: %lu \n", size(pred_result), size(pred_result[0]));
    // printRawResult(pred_result);
    
    /* process result */
    auto result = reshapeResult(pred_result[0], outputSize); // extract single output & reshape
    vector<tuple<double, double>> classCoords = processResult(result, outputSize, THRESHOLD_CONFIDENCE);
    // classCoords now contain most confident coords of each 18 significant classes,
    // if coord of a class is (-1,-1) then the class was not detected in image

    /* render result */
    track(classCoords, WIDTH, HEIGHT); // resize & store coords in points
    drawPose(points, canvas, img_RGB, "Image"); // draw points and lines

    return 0;
}