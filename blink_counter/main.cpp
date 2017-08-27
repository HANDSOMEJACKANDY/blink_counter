//
//  main.cpp
//  blink_counter
//
//  Created by AndyWu on 17/08/2017.
//  Copyright © 2017 AndyWu. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <fstream>

using namespace std;
using namespace cv;

string inputDir = "blink_counter/haarcascades/";
String face_cascade_name = inputDir + "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = inputDir + "haarcascade_lefteye_2splits.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
Mat curFrame, prevFrame, colorFrame;
// for optic flow:
vector<Point2f> point[2]; // point0为特征点的原来位置，point1为特征点的新位置
vector<Point2f> initPoint;    // 初始化跟踪点的位置
vector<Point2f> features; // 检测的特征
int maxCount = 2000;         // 检测的最大特征数
double qLevel = 0.01;   // 特征检测的等级
double minDist = 10.0;  // 两特征点之间的最小距离
vector<uchar> status; // 跟踪特征的状态，特征的流发现为1，否则为0
vector<float> err;

void thresholdAndOpen(Mat &src, Mat &dst); //threshold and consequent bluring operation
void thresholdInRange(Mat &src, Mat &dst); //figure out the range of the eye_blink_diff
vector<Rect> detectEyeAndFace(Mat src);
// optic flow tracking
void opticalFlow(Mat &frame, Mat &result);
bool addNewPoints();
bool acceptTrackedPoint(int i);

int main(){
    VideoCapture cap;
    cap.open(0);
    
    if(!cap.isOpened()){
        cout << "fail to open camera" << endl;
        return -1;
    }
    
    Mat residue;
    namedWindow("camera", 1);
    namedWindow("residue", 1);
    cap >> prevFrame;
    pyrDown(prevFrame, prevFrame);
    prevFrame.copyTo(colorFrame);
    cvtColor(prevFrame, prevFrame, COLOR_BGR2GRAY);
    cout << prevFrame.size[0] << endl << prevFrame.size[1] << endl;
    
//  init face and eye casacade
    if( !face_cascade.load( face_cascade_name ) ){
        printf("face_cascade_name加载失败\n");
        getchar();
    }
    if( !eyes_cascade.load( eyes_cascade_name ) ){
        printf("eye_cascade_name加载失败\n");
        getchar();
    }
    
//  main process
    bool is_tracking = false;
    Rect box;
    
    while(waitKey(1) != 27){
        double start = getTickCount(); //to count the period the process takes
        
        cap >> colorFrame;
        pyrDown(colorFrame, colorFrame);
        colorFrame.copyTo(curFrame);
        cvtColor(curFrame, curFrame, COLOR_BGR2GRAY);
        residue = curFrame - prevFrame;
        
        //thresholdAndOpen(residue, residue);
        //thresholdInRange(residue, residue);
        
//        if(!is_tracking){
            vector<Rect> eyes = detectEyeAndFace(curFrame);
//            if(eyes.size()){
//                box = eyes[0];
//                tracker->init(frame[i], box);
//                is_tracking = true;
//            }
//        }

        
        
//        rectangle(frame[i], box, 255, 3);
        
        opticalFlow(curFrame, colorFrame);
        
        for(size_t j=0; j<eyes.size(); j++)
            rectangle(colorFrame, eyes[j], Scalar(255, 0, 0), 3);
        
        imshow("camera", colorFrame);
        imshow("residue", residue);
        
//        swap prev and cur
        curFrame.copyTo(prevFrame);
        swap(point[1], point[0]);
        
        cout << ((getTickCount() - start) / getTickFrequency()) * 1000 << endl; //output the time the process takes
    }
    
    return 0;
}

void thresholdAndOpen(Mat &src, Mat &dst){
    threshold(src, dst, 10, 255, THRESH_BINARY);
    int maskSize = 5;
    //threshold(residue, residue, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, maskSize, 0);
    Mat ele = getStructuringElement(MORPH_ELLIPSE, Size(maskSize, maskSize));
    morphologyEx(src, dst, MORPH_OPEN, ele);
}

void thresholdInRange(Mat &src, Mat &dst){
    inRange(src, 10, 30, dst);
    int maskSize = 5;
    Mat ele = getStructuringElement(MORPH_ELLIPSE, Size(maskSize, maskSize));
    morphologyEx(src, dst, MORPH_OPEN, ele);
}

vector<Rect> detectEyeAndFace(Mat src){
    vector<Rect> faces, eyes;
    
    face_cascade.detectMultiScale( src, faces, 1.1, 2, 0, Size(30, 30));
    
    for( size_t i = 0; i < faces.size(); i++)
    {
        Mat faceROI = src( faces[i] );
        
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0, Size(30, 30));
        
        for( size_t j = 0; j < eyes.size(); j++)
            eyes[j] += faces[i].tl();
    }
    
    return eyes;
}

void opticalFlow(Mat &src, Mat &dst)
{
    if (addNewPoints())
    {
        goodFeaturesToTrack(curFrame, features, maxCount, qLevel, minDist);
        point[0].insert(point[0].end(), features.begin(), features.end());
        initPoint.insert(initPoint.end(), features.begin(), features.end());
    }
    
    calcOpticalFlowPyrLK(prevFrame, curFrame, point[0], point[1], status, err);
    
    int k = 0;
    for (size_t i = 0; i<point[1].size(); i++)
    {
        if (acceptTrackedPoint(int(i)))
        {
            initPoint[k] = initPoint[i];
            point[1][k++] = point[1][i];
        }
    }
    
    
    point[1].resize(k);
    initPoint.resize(k);
    
    for (size_t i = 0; i<point[1].size(); i++)
    {
        line(dst, initPoint[i], point[1][i], Scalar(0, 0, 255));
        circle(dst, point[1][i], 3, Scalar(0, 255, 0), -1);
    }
}


bool addNewPoints()
{
    return point[0].size() <= 10;
}


bool acceptTrackedPoint(int i)
{
    return status[i] && ((abs(point[0][i].x - point[1][i].x) + abs(point[0][i].y - point[1][i].y)) > 0.1);
}







