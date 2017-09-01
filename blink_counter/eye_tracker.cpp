//
//  eye_tracker.cpp
//  blink_counter
//
//  Created by AndyWu on 31/08/2017.
//  Copyright © 2017 AndyWu. All rights reserved.
//

#include "eye_tracker.hpp"

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

EyeTracker::EyeTracker(){
    //  init face and eye casacade
    if( !face_cascade.load( face_cascade_name ) ){
        printf("face_cascade_name加载失败\n");
        getchar();
    }
    if( !eyes_cascade.load( eyes_cascade_name ) ){
        printf("eye_cascade_name加载失败\n");
        getchar();
    }
}

Rect EyeTracker::tracking(double inputScale){
    if(curFrame.empty()){
        originFrame.copyTo(curFrame);
        inputScale = rescale(curFrame, curFrame, inputScale);
        prevFrame.copyTo(colorFrame);
        cvtColor(curFrame, curFrame, COLOR_BGR2GRAY);
        scale = inputScale;
    }
    else{
        Rect eye;

        double start = getTickCount(); //to count the period the process takes
        
        //swap prev and cur
        curFrame.copyTo(prevFrame);
        swap(point[1], point[0]);
        //get current frame
        originFrame.copyTo(colorFrame);
        inputScale = rescale(colorFrame, colorFrame, inputScale);
        colorFrame.copyTo(curFrame);
        cvtColor(curFrame, curFrame, COLOR_BGR2GRAY);
        
        // check if the scaling of prev frame is identical
        if(scale != inputScale){
            rescale(prevFrame, prevFrame, inputScale / scale);
            for(size_t i=0; i<point[0].size(); i++){
                point[0][i] *= inputScale / scale;
                point[1][i] *= inputScale / scale;
                initPoint[i] *= inputScale / scale;
            }
            tbCenter *= inputScale / scale;
            tbWidth *= inputScale / scale;
            tbHeight *= inputScale / scale;
            trackingBox = getTrackingBox();
            
            scale = inputScale;
        }
        
        if(!is_tracking){
            if(findMostRightEye(detectEyeAndFace(curFrame), eye)){
                trackingBox = eye;
                //recording the info of tracking box
                tbWidth = trackingBox.width;
                tbHeight = trackingBox.height;
                tbCenter = trackingBox.tl() + Point(tbWidth / 2, tbHeight / 2);
                is_tracking = true;
            }
        }
        else{
            if(!enlargedRect(trackingBox, 2.5).empty()){
                opticalFlow(enlargedRect(trackingBox, 2.5));
                if(point[1].size() != 0){
                    tbCenter += Point(filteredDisplacement());
                    trackingBox = getTrackingBox();
                }
            }
            else
                is_tracking = false;
        }
        
        // cout << "instant time: " << ((getTickCount() - start) / getTickFrequency()) * 1000 << endl;
        sumTime += ((getTickCount() - start) / getTickFrequency()) * 1000; //output the time the process takes
        timeCount++;
        
        originTrackingBox = Rect(trackingBox.tl() / scale, trackingBox.br() / scale);
    }
    
    return originTrackingBox;
}

double EyeTracker::rescale(Mat src, Mat &dst, double inputScale){
    src.copyTo(dst);
    double tempScale = 1;
    if(inputScale > 1){
        for(; inputScale > 1; inputScale /= 2){
            pyrUp(dst, dst);
            tempScale *= 2;
        }
    }
    else if(inputScale <= 1 && inputScale > 0){
        for(; inputScale < 1; inputScale *= 2){
            pyrDown(dst, dst);
            tempScale /= 2;
        }
    }
    else
        tempScale = 1;
    if(inputScale != 1){
        cout << "inputScale is not the power of 2" << endl;
        cout << "So we use scale: " << tempScale << " instead" << endl;
    }
    else
        cout << "rescaling successful" << endl;
    
    return tempScale;
}

vector<Rect> EyeTracker::detectEyeAndFace(Mat src, bool isFace){
    vector<Rect> eyes;
    if(isFace){
        vector<Rect> faces;
        
        face_cascade.detectMultiScale( src, faces, 1.1, 2, 0, Size(30, 30));
        
        for( size_t i = 0; i < faces.size(); i++)
        {
            Mat faceROI = src( faces[i] );
            
            eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0, Size(30, 30));
            
            for( size_t j = 0; j < eyes.size(); j++)
                eyes[j] += faces[i].tl();
        }
    }
    else{
        eyes_cascade.detectMultiScale( src, eyes, 1.1, 2, 0, Size(30, 30));
    }
    
    return eyes;
}

bool EyeTracker::findMostRightEye(vector<Rect> eyes, Rect &eye){
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

Rect EyeTracker::getTrackingBox(){
    return enlargedRect(Rect(Point(tbCenter.x - tbWidth / 2, tbCenter.y - tbHeight / 2), Point(tbCenter.x + tbWidth / 2, tbCenter.y + tbHeight / 2)), 1);
}

void EyeTracker::drawTrackingBox(Mat &dst){
    if(dst.empty())
        originFrame.copyTo(dst);
    
    rectangle(dst, originTrackingBox, Scalar(0, 0, 255), 3);
    
}

Rect EyeTracker::enlargedRect(Rect src, float times){
    Point tl, br;
    tl.x = max(src.tl().x - src.width * (times - 1) / 2, 0);
    tl.x = min(tl.x, curFrame.size().width);
    tl.y = max(src.tl().y - src.height * (times - 1) / 2, 0);
    tl.y = min(tl.y, curFrame.size().height);
    br.x = min(src.br().x + src.width * (times - 1) / 2, curFrame.size().width);
    br.x = max(br.x, 0);
    br.y = min(src.br().y + src.height * (times - 1) / 2, curFrame.size().height);
    br.y = max(br.y, 0);
    return Rect(tl, br);
}

void EyeTracker::opticalFlow(Rect src)
{
    if (addNewPoints())
    {   // first extract features in trackingBox by centerFeaturePercentage
        goodFeaturesToTrack(curFrame(trackingBox), features, (maxCount - int(point[0].size())) * centerFeaturePercentage, qLevel, minDist);
        for(size_t i=0; i<features.size(); i++){
            features[i] += Point2f(trackingBox.tl());
        }
        point[0].insert(point[0].end(), features.begin(), features.end());
        initPoint.insert(initPoint.end(), features.begin(), features.end());
        // then we extract features outside the trackingBox
        if(centerFeaturePercentage < 1){
            goodFeaturesToTrack(curFrame(src), features, (maxCount - int(point[0].size())), qLevel, minDist);
            for(vector<Point2f>::iterator iter=features.begin(); iter != features.end();){
                *iter += Point2f(src.tl());
                if((*iter).inside(trackingBox))
                    features.erase(iter);
                else
                    iter++;
            }
            point[0].insert(point[0].end(), features.begin(), features.end());
            initPoint.insert(initPoint.end(), features.begin(), features.end());
        }
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
        if (status[i] && ((abs(tempPoint[0][i].x - tempPoint[1][i].x) + abs(tempPoint[0][i].y - tempPoint[1][i].y)) > 0.1)) // check if the point is qualified
        {
            initPoint[k] = initPoint[i];
            point[1][k] = tempPoint[1][i] + Point2f(src.tl());
            point[0][k++] = tempPoint[0][i] + Point2f(src.tl());
        }
    }
    
    point[0].resize(k);
    point[1].resize(k);
    initPoint.resize(k);
}

bool EyeTracker::addNewPoints()
{
    return point[0].size() <= minCount;
}

void EyeTracker::drawOptFlow(Mat &dst){
    if(dst.empty())
        originFrame.copyTo(dst);
    
    for (size_t i = 0; i<point[1].size(); i++)
    {
        line(dst, initPoint[i] / scale, point[1][i] / scale, Scalar(0, 0, 255));
        circle(dst, point[1][i] / scale, 3, Scalar(0, 255, 0), -1);
    }
}

Point2f EyeTracker::filteredDisplacement(){
    size_t size(point[0].size());
    float disX(0), disY(0);
    
    vector<DisFilter> tempDis;
    tempDis.resize(size);
    for(size_t i=0; i<size; i++){
        tempDis[i].dis = point[1][i] - point[0][i];
        tempDis[i].flag = false;
        tempDis[i].seq = int(i);
    }
    
    vector<DisFilter>::iterator iter;
    sort(tempDis.begin(), tempDis.end(), compX);
    iter = tempDis.begin();
    for(int i=0; i<size*filterPercentage; i++){
        (iter++)->flag = true;
    }
    iter = tempDis.end();
    for(int i=0; i<size*filterPercentage; i++){
        (--iter)->flag = true;
    }
    
    sort(tempDis.begin(), tempDis.end(), compY);
    iter = tempDis.begin();
    for(int i=0; i<size*filterPercentage; i++){
        (iter++)->flag = true;
    }
    iter = tempDis.end();
    for(int i=0; i<size*filterPercentage; i++){
        (--iter)->flag = true;
    }
    
//    sort(tempDis.begin(), tempDis.end(), compYX);
//    iter = tempDis.begin();
//    for(int i=0; i<size*filterPercentage; i++){
//        (iter++)->flag = true;
//    }
//    iter = tempDis.end();
//    for(int i=0; i<size*filterPercentage; i++){
//        (--iter)->flag = true;
//    }
    
    for(iter=tempDis.begin(); iter != tempDis.end();){
        if(iter->flag == true){
            *(point[0].begin() + iter->seq) = Point2f(-1, -1);
            *(point[1].begin() + iter->seq) = Point2f(-1, -1);
            tempDis.erase(iter);
        }
        else
            iter++;
    }
    
    for(size_t i =0; i<point[0].size();){
        if(point[0][i] == Point2f(-1, -1)){
            point[0].erase(point[0].begin() + i);
            point[1].erase(point[1].begin() + i);
            initPoint.erase(initPoint.begin() + i);
        }
        else
            i++;
    }
    
    for(size_t i=0; i<tempDis.size(); i++){
        disX += tempDis[i].dis.x;
        disY += tempDis[i].dis.y;
    }
    disX /= tempDis.size();
    disY /= tempDis.size();
    
    return Point2f(disX, disY);
}

bool EyeTracker::compX(const DisFilter a, const DisFilter b){
    return a.dis.x < b.dis.x;
}

bool EyeTracker::compY(const DisFilter a, const DisFilter b){
    return a.dis.y < b.dis.y;
}

bool EyeTracker::compYX(const DisFilter a, const DisFilter b){
    return a.dis.y / a.dis.x < b.dis.y / b.dis.x;
}

double EyeTracker::getAverageTime(){
    return sumTime / timeCount;
}
