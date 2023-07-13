#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

//global variables needed for poseView

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

void renderPose(string windowName, Mat image){
    canvas = Mat(Size(1000,1000), CV_8UC3, Scalar(255, 255, 255));
    points.resize(TOTAL_POINTS);
}

/* Record latest data, which will be shown in the next call of onDraw(). */
void track(vector<tuple<int,int>> nodePosition) {
        for (int i = 0; i < TOTAL_POINTS; ++i) {
            if (i < nodePosition.size() && get<0>(nodePosition[i]) >= 0) {
                // Resize output based on size of the view.
                //need to get DIM_OUTPUT_W of PoseDetector and width of the screen
                //need to get DIM_OUTPUT_W of PoseDetector and width of the screen
                int DIM_OUTPUT_W = 200, DIM_OUTPUT_H = 200;
                int width = 1000, height = 1000;
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
}

void drawImgBox(string windowName, Mat image, Point topLeft, Point botRight){
    // Create a black screen
    Mat screen(Size(1000,1000), CV_8UC3, Scalar(0, 0, 0));
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

int main(){
    //Create a window to display the image
    namedWindow("Image");
    Mat image = imread("./../yuzuru.jpg");
    Mat resized_image;
    resize(image, resized_image, Size(224,224), INTER_LINEAR);
    drawImgBox("Image", resized_image, Point(20, 60),Point(100, 100) );
    
}