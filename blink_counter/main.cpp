//
//  main.cpp
//  blink_counter
//
//  Created by AndyWu on 17/08/2017.
//  Copyright © 2017 AndyWu. All rights reserved.
//

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
#include <fstream>

using namespace std;
using namespace cv;

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

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
double qLevel = 0.001;   // 特征检测的等级
double minDist = 10.0;  // 两特征点之间的最小距离
vector<uchar> status; // 跟踪特征的状态，特征的流发现为1，否则为0
vector<float> err;

void thresholdAndOpen(Mat &src, Mat &dst); //threshold and consequent bluring operation
void thresholdInRange(Mat &src, Mat &dst); //figure out the range of the eye_blink_diff
vector<Rect> detectEyeAndFace(Mat src);
bool findMostRightEye(vector<Rect> eyes, Rect &eye);
Rect enlargedRect(Rect src, float times);
// optic flow tracking
void opticalFlow(Rect src, Mat &dst); //tracking ROI
bool addNewPoints();
Point2f filteredDisplacement();
bool compX(const Point2f a, const Point2f b);
bool compY(const Point2f a, const Point2f b);

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
    Rect eye, trackingBox;
    
    while(waitKey(1) != 27){
        double start = getTickCount(); //to count the period the process takes
        
        cap >> colorFrame;
        pyrDown(colorFrame, colorFrame);
        colorFrame.copyTo(curFrame);
        cvtColor(curFrame, curFrame, COLOR_BGR2GRAY);
        residue = curFrame - prevFrame;
        
        //thresholdAndOpen(residue, residue);
        //thresholdInRange(residue, residue);
        
        if(!is_tracking){
            if(findMostRightEye(detectEyeAndFace(curFrame), eye)){
                trackingBox = enlargedRect(eye, 2);
                rectangle(colorFrame, trackingBox, Scalar(0, 255, 0), 3); // draw eyes onto the colorframe
                is_tracking = true;
            }
        }
        else{
            if(!trackingBox.empty()){
                opticalFlow(trackingBox, colorFrame);
                trackingBox += Point(filteredDisplacement()); // may get out of the screen……
            }
            else
                is_tracking = false;
        }

        rectangle(colorFrame, trackingBox, Scalar(255, 0, 0), 3);
        
        
        imshow("camera", colorFrame);
        imshow("residue", residue);
        
//        swap prev and cur
        curFrame.copyTo(prevFrame);
        
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

bool findMostRightEye(vector<Rect> eyes, Rect &eye){
    int index(-1), x(0);
    for(size_t i=0; i<eyes.size(); i++){
        if(eyes[i].br().x + eyes[i].width / 2 > x){
            x = eyes[i].br().x + eyes[i].width / 2;
            index = int(i);
        }
    }
    
    if(index == -1){
        return false;
    }
    else{
        eye = eyes[index];
        return true;
    }
}

Rect enlargedRect(Rect src, float times){
    Point tl, br;
    tl.x = max(src.tl().x - src.width * (times - 1) / 2, 0);
    tl.y = max(src.tl().y - src.height * (times - 1) / 2, 0);
    br.x = min(src.br().x + src.width * (times - 1) / 2, curFrame.size().width);
    br.y = min(src.br().y + src.height * (times - 1) / 2, curFrame.size().height);
    return Rect(tl, br);
}

void opticalFlow(Rect src, Mat &dst)
{
    if (addNewPoints())
    {
        goodFeaturesToTrack(curFrame(src), features, maxCount, qLevel, minDist);
        for(size_t i=0; i<features.size(); i++){
            features[i] += Point2f(src.tl());
        }
        point[0].insert(point[0].end(), features.begin(), features.end());
        initPoint.insert(initPoint.end(), features.begin(), features.end());
    }
    
    vector<Point2f> tempPoint[2];
    tempPoint[0].resize(point[0].size());
    for(size_t i=0; i<tempPoint[0].size(); i++){
        tempPoint[0][i] = point[0][i] - Point2f(src.tl());
    }
    
    calcOpticalFlowPyrLK(prevFrame(src), curFrame(src), tempPoint[0], tempPoint[1], status, err);
    point[1].resize(tempPoint[1].size());
    
    int k = 0;
    for (size_t i = 0; i<tempPoint[1].size(); i++)
    {
        if (status[i] && ((abs(tempPoint[0][i].x - tempPoint[1][i].x) + abs(tempPoint[0][i].y - tempPoint[1][i].y)) > 0.1))
        {
            initPoint[k] = initPoint[i];
            point[1][k] = tempPoint[1][i] + Point2f(src.tl());
            point[0][k++] = tempPoint[0][i] + Point2f(src.tl());
        }
    }
    
    point[0].resize(k);
    point[1].resize(k);
    initPoint.resize(k);
    
    for (size_t i = 0; i<point[1].size(); i++)
    {
        line(dst, initPoint[i], point[1][i], Scalar(0, 0, 255));
        circle(dst, point[1][i], 3, Scalar(0, 255, 0), -1);
    }
    
    swap(point[1], point[0]);
}

bool addNewPoints()
{
    return point[0].size() <= 20;
}

Point2f filteredDisplacement(){
    size_t size(point[0].size());
    float disX(0), disY(0);
    vector<Point2f> dis;
    dis.resize(size);
    for(size_t i=0; i<size; i++){
        dis[i] = point[0][i] - point[1][i];
    }
//    sort(dis.begin(), dis.end(), compX);
//    for(vector<Point2f>::iterator iter=dis.end(); iter!=dis.end() - size/20; iter--){
//        dis.erase(iter);
//    }
//    for(vector<Point2f>::iterator iter=dis.begin(); iter!=dis.begin() + size/20; iter++){
//        dis.erase(iter);
//    }
//    sort(dis.begin(), dis.end(), compX);
//    for(vector<Point2f>::iterator iter=dis.end(); iter!=dis.end() - size/20; iter--){
//        dis.erase(iter);
//    }
//    for(vector<Point2f>::iterator iter=dis.begin(); iter!=dis.begin() + size/20; iter++){
//        dis.erase(iter);
//    }
    for(size_t i=0; i<dis.size(); i++){
        disX += dis[i].x;
        disY += dis[i].y;
    }
    disX /= dis.size();
    disY /= dis.size();
    return Point2f(disX, disY);
}

bool compX(const Point2f a, const Point2f b){
    return a.x < b.x;
}

bool compY(const Point2f a, const Point2f b){
    return a.y < b.y;
}





