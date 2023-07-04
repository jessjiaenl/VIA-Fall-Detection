#include "neuropl.cpp"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;


/* global variables needed for rendering */
const int SCREEN_SIZE = 1000;
const double STRIKE_WIDTH_TEXT = 8.0;
const double STRIKE_WIDTH_LINE = 3.0;
const int TOTAL_POINTS = 18;
//format a r g b, each is 8 bits
const int COLOR_HEAD = 0xB40000FF; 
const int COLOR_BODY = 0xB4FFFF00;
const int COLOR_FOOT = 0xB400FF00;
const int COLOR_BACKGROUND = 0xB4FFFFFF;
const int COLOR_LINE = 0xB4888888;
const int COLOR_TEXT = 0xB4FF0000;

const vector<int> POINT_COLORS = {
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
    COLOR_BACKGROUND  // 18. background
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
vector<tuple<int,int>> points;
Mat canvas;
/* end of global vars */

// initialize canvas
void renderPose(Mat image){
    canvas = Mat(Size(SCREEN_SIZE,SCREEN_SIZE), CV_8UC3, Scalar(255, 255, 255));
    points.resize(TOTAL_POINTS);
}

/* Record latest data, which will be shown in the next call of onDraw(). */
void track(vector<tuple<int,int>> nodePosition, int w, int h) {
        for (int i = 0; i < TOTAL_POINTS; ++i) {
            if (i < nodePosition.size() && get<0>(nodePosition[i]) >= 0) {
                // Resize output based on size of the view.
                //need to get DIM_OUTPUT_W of PoseDetector and width of the screen
                //need to get DIM_OUTPUT_W of PoseDetector and width of the screen
                int DIM_OUTPUT_W = w, DIM_OUTPUT_H = h;
                int width = SCREEN_SIZE, height = SCREEN_SIZE;
                points[i] = make_tuple(get<0>(nodePosition[i])/ DIM_OUTPUT_W * width, get<1>(nodePosition[i]) / DIM_OUTPUT_H * height);
            } else {
                points[i] = make_tuple(-1, -1); //-1 is used as a placeholder for NULL
            }
        }
    }

void drawPose(const vector<tuple<int, int>>& points, Mat& canvas) {
    // Draw points.
    for (int i = 0; i < TOTAL_POINTS; ++i) {
        tuple<int, int> point = points[i];
        if (get<0>(point) >= 0 && get<1>(point) >= 0) {
            int color = POINT_COLORS[i];
            bool isFacePoint = find(faceIndex.begin(), faceIndex.end(), to_string(i)) != faceIndex.end();
            //convert tuple into cv::point
            Point p = Point(get<0>(point), get<1>(point));
            circle(canvas, p, (isFacePoint ? 8 : 12), color, FILLED);
            int textColor = COLOR_TEXT;
            float textSize = isFacePoint ? 20.0f : 30.0f;
            putText(canvas, to_string(i), p, FONT_HERSHEY_SIMPLEX, textSize, textColor, 2);
        }
    }

    // Draw lines.
    for (const auto& linePoint : LINE_POINTS) {
        int startIndex = get<0>(linePoint);
        int endIndex =  get<1>(linePoint);
        tuple<int, int> startPoint = points[startIndex];
        tuple<int, int> endPoint = points[endIndex];
        if ( get<0>(startPoint) >= 0 &&  get<1>(startPoint) >= 0 &&  get<0>(endPoint) >= 0 && get<1>(endPoint) >= 0) {
            //convert tuple to point
            Point start = Point(get<0>(startPoint), get<1>(startPoint));
            Point end = Point(get<0>(endPoint), get<1>(endPoint));
            line(canvas, start, end, COLOR_LINE, STRIKE_WIDTH_LINE);
        }
    }
    imshow("Image", canvas);
    //Wait for a keyboard event and handle it:
    waitKey(0);
    //Destroy the window and release resources:
    destroyWindow("Image");
}

void printRawResult(auto& result){
    for (auto output : result) {
            for (auto item : output) {
                printf("%d ", item);
            }
            printf("\n");
        }
}

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

int main(void){
    namedWindow("Image");
    // edit model & test parameters here:
    string model_path = "./../openpose_mobilenetv0.75_quant_1x368x368x3.dla";
    vector<int> outputSize = {1,46,46,57};
    int frame_w = 368, frame_h = 368;
    string test_img_path = "./../yuzuru.jpg";
    float OUTPUT_RATIO = 0.008926701731979847;
    float OUTPUT_BIAS = 126;
    float THRESHOLD_CONFIDENCE = 0.2/(OUTPUT_RATIO+OUTPUT_BIAS);

    // initialize objects
    Neuropl model {model_path};
    Mat img = imread(test_img_path);
    renderPose(img); // init render

    // crop and resize img
    auto w = img.size[1], h = img.size[0]; // type int
    auto target_len = min(w, h);
    auto start_x = (w-target_len)/2, start_y = (h-target_len)/2;
    Mat cropped = img(Range(start_y, start_y+target_len), Range(start_x, start_x+target_len));
    Mat resized;
    resize(cropped, resized, Size(frame_w, frame_h));
    Mat img_RGB;
    cvtColor(resized, img_RGB, COLOR_BGR2RGB);

    // predict
    uint8_t* byte_buffer = reinterpret_cast<uint8_t*>(img_RGB.data);
    auto pred_result = model.predict(byte_buffer);

    // print raw result
    // printRawResult(pred_result);

    // reshape result
    auto result = reshapeResult(pred_result[0], outputSize); // extract single output & reshape

    // process result
    vector<tuple<int, int>> classCoords(18, make_tuple(-1,-1));
    vector<float> classConfidences(18, THRESHOLD_CONFIDENCE);
    for (int i = 0; i < outputSize[1]; i++){
        for (int j = 0; j < outputSize[2]; j++){
            vector<uint8_t>::iterator maxConfPtr = max_element(result[0][i][j].begin(), result[0][i][j].end());
            uint8_t maxClass = distance(result[0][i][j].begin(), maxConfPtr);
            if (maxClass < 18){
                if ((*maxConfPtr) >= classConfidences[maxClass]){
                    classCoords[maxClass] = make_tuple(i,j);
                    classConfidences[maxClass] = (*maxConfPtr);
                }
            }
        }
    }
    // classCoords now contain most confident coords of each 18 significant classes,
    // if coord of a class is (-1,-1) then the class was not detected in image
    for (auto output : classConfidences) {
            printf("%d ", output);
            printf("\n");
        }

    // render result image
    track(classCoords, frame_w, frame_h);
    drawPose(classCoords, canvas);

    return 0;
}