//
//  eye_tracker.hpp
//  blink_counter
//
//  Created by AndyWu on 31/08/2017.
//  Copyright © 2017 AndyWu. All rights reserved.
//

#ifndef eye_tracker_hpp
#define eye_tracker_hpp

#include <stdio.h>

#endif /* eye_tracker_hpp */

#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <time.h>

using namespace std;
using namespace cv;

class EyeTracker{
public:
    EyeTracker();
    
public:
    void trackByScale(double inputScale);
    bool tuneByDetection(double step, double trackRegionScale = 1);
    void checkIsTracking();
    
private:
    struct DisFilter{ //to filt the point that may be falsely matched
        Point2f dis;
        int seq;
        bool flag;
    };
    
public:
    // for image processing
    double rescale(Mat src, Mat &dst, double inputScale);
    // for eye detection
    vector<Rect> detectEyeAndFace(Mat src, bool isFace = true);
    bool findMostRightEye(vector<Rect> eyes, Rect &eye);
    // for tracking box
    Rect enlargedRect(Rect src, float times, bool isDefault = true);
    void getTrackingBox();
    void drawTrackingBox(Mat &dst);
    void kMeansTuning(vector<Point2f> &eyeCenters);
    // optic flow tracking
    void opticalFlow(Rect src); //tracking ROI
    bool addNewPoints();
    void drawOptFlow(Mat &dst);
    // filter points
    Point2f filteredDisplacement();
    static bool compX(const DisFilter a, const DisFilter b);
    static bool compY(const DisFilter a, const DisFilter b);
    static bool compYX(const DisFilter a, const DisFilter b);
    
    // for efficiency evaluation
    void setTimeStart();
    void setTimeEnd();
    double getAverageTime();
    
public:
    string inputDir = "blink_counter/haarcascades/";
    String face_cascade_name = inputDir + "haarcascade_frontalface_alt.xml";
    String eyes_cascade_name = inputDir + "haarcascade_lefteye_2splits.xml";
    CascadeClassifier face_cascade;
    CascadeClassifier eyes_cascade;
    Mat curFrame, prevFrame, colorFrame, originFrame;
    double scale = 2;
    // for tracking box:
    bool is_tracking = false;
    Rect trackingBox = Rect(0,0,0,0), originTrackingBox = Rect(0,0,0,0);
    Point tbCenter = Point(0,0);
    int tbWidth = 0, tbHeight = 0;
    double tbAngle = 0;
    // for checkistracking
    int maxLostFrame = 10;
    // for tune by detection
    double tuningPercentage = 0.25, rectForTrackPercentage = 1.75;
    float lostFrame = 0;
    bool isLostFrame = false;
    // for optic flow:
    vector<Point2f> point[2]; // point0为特征点的原来位置，point1为特征点的新位置
    vector<Point2f> initPoint;    // 初始化跟踪点的位置
    vector<Point2f> features; // 检测的特征
    int maxCount = 100, minCount = 50;         // 检测的最大特征数
    float centerFeaturePercentage = 0.666;
    double qLevel = 0.01;   // 特征检测的等级
    double minDist = 10.0;  // 两特征点之间的最小距离
    vector<uchar> status; // 跟踪特征的状态，特征的流发现为1，否则为0
    vector<float> err;
    // filtering displacement
    float filterPercentage = 0.15; //seems perfect with 1/5
    // for time counting
    clock_t start;
    double sumTime = 0, timeCount = 0;
};
