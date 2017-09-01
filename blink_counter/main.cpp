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
#include "eye_tracker.hpp"

using namespace std;
using namespace cv;


int main(){
    VideoCapture cap;
    Mat frame;
    EyeTracker tracker;
    
    cap.open(0);
    
    if(!cap.isOpened()){
        cout << "fail to open camera" << endl;
        return -1;
    }
    
    namedWindow("curFrame");
    namedWindow("camera");
    Rect trackingBox;
    while(waitKey(1) != 27){
        cap >> frame;
        frame.copyTo(tracker.originFrame);
        trackingBox = tracker.tracking(0.5);
        tracker.drawTrackingBox(frame);
        tracker.drawOptFlow(frame);
        imshow("camera", frame);
        imshow("curFrame", tracker.curFrame);
         cout << tracker.getAverageTime() << endl;
    }
    
    return 0;
}
