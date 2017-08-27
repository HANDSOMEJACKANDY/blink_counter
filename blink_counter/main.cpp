//
//  main.cpp
//  blink_counter
//
//  Created by AndyWu on 17/08/2017.
//  Copyright © 2017 AndyWu. All rights reserved.
//

#include <iostream>
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

void thresholdAndOpen(Mat src, Mat dst); //threshold and consequent bluring operation
void thresholdInRange(Mat src, Mat dst); //figure out the range of the eye_blink_diff
vector<Rect> detectEyeAndFace(Mat src);

int main(){
    VideoCapture cap;
    cap.open(0);
    
    if(!cap.isOpened()){
        cout << "fail to open camera" << endl;
        return -1;
    }
    
    Mat frame[2], residue;
    namedWindow("camera", 1);
    namedWindow("residue", 1);
    cap >> frame[1];
    cvtColor(frame[1], frame[1], COLOR_BGR2GRAY);
    pyrDown(frame[1], frame[1]);
    cout << frame[1].size[0] << endl << frame[1].size[1] << endl;
    
//  init face and eye casacade
    if( !face_cascade.load( face_cascade_name ) ){
        printf("face_cascade_name加载失败\n");
        getchar();
    }
    if( !eyes_cascade.load( eyes_cascade_name ) ){
        printf("eye_cascade_name加载失败\n");
        getchar();
    }

// create tracker
//    Ptr<Tracker> tracker = Tracker::create();
    
    
    
//  main process
    bool is_tracking = false;
    Rect box;
    
    for(int i=0; waitKey(1) != 27; i=!i){
        double start = getTickCount(); //to count the period the process takes
        
        cap >> frame[i];
        cvtColor(frame[i], frame[i], COLOR_BGR2GRAY);
        pyrDown(frame[i], frame[i]);
        //equalizeHist(frame[i], frame[i]);
        residue = frame[i] - frame[!i];
        
        //thresholdAndOpen(residue, residue);
        //thresholdInRange(residue, residue);
//        if(!is_tracking){
            vector<Rect> eyes = detectEyeAndFace(frame[i]);
//            if(eyes.size()){
//                box = eyes[0];
//                tracker->init(frame[i], box);
//                is_tracking = true;
//            }
//        }

        
        
//        rectangle(frame[i], box, 255, 3);
        for(size_t j=0; j<eyes.size(); j++)
            rectangle(frame[i], eyes[j], 255, 3);

        imshow("camera", frame[i]);
        imshow("residue", residue);
        
        cout << ((getTickCount() - start) / getTickFrequency()) * 1000 << endl; //output the time the process takes
    }
}

void thresholdAndOpen(Mat src, Mat dst){
    threshold(src, dst, 10, 255, THRESH_BINARY);
    int maskSize = 5;
    //threshold(residue, residue, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, maskSize, 0);
    Mat ele = getStructuringElement(MORPH_ELLIPSE, Size(maskSize, maskSize));
    morphologyEx(src, dst, MORPH_OPEN, ele);
}

void thresholdInRange(Mat src, Mat dst){
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









