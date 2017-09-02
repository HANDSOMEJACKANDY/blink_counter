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

void EyeTracker::trackByScale(double inputScale){
    if(curFrame.empty()){
        originFrame.copyTo(curFrame);
        inputScale = rescale(curFrame, curFrame, inputScale);
        prevFrame.copyTo(colorFrame);
        cvtColor(curFrame, curFrame, COLOR_BGR2GRAY);
        scale = inputScale;
    }
    else{
        Rect eye;

        // double start = getTickCount(); //to count the period the process takes

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
            getTrackingBox();
            
            scale = inputScale;
        }
        // check if tracking is failed
        checkIsTracking();

        if(!is_tracking){
            if(findMostRightEye(detectEyeAndFace(curFrame), eye)){
                trackingBox = eye;
                //recording the info of tracking box
                tbWidth = trackingBox.width;
                tbHeight = trackingBox.height;
                tbCenter = trackingBox.tl() + Point(tbWidth / 2, tbHeight / 2);
                is_tracking = true;
            }
            else{
                tbWidth = -1;
                tbHeight = -1;
                tbCenter = Point2f(-1, -1);
                point[0].resize(0);
                point[1].resize(0);
                initPoint.resize(0);
            }
        }
        else{
            if(!enlargedRect(trackingBox, 2.5).empty()){
                opticalFlow(enlargedRect(trackingBox, 2.5));
                if(point[1].size() != 0){
                    tbCenter += Point(filteredDisplacement());
                    getTrackingBox();
                }
            }
            else
                is_tracking = false;
        }
        
        //cout << "instant time: " << ((getTickCount() - start) / getTickFrequency()) * 1000 << endl;
        //sumTime += ((getTickCount() - start) / getTickFrequency()) * 1000; //output the time the process takes
    }
}

bool EyeTracker::tuneByDetection(double step, double trackRegionScale){
    if(enlargedRect(originTrackingBox,trackRegionScale, false).empty())
        return false;
    
    namedWindow("tune");
    namedWindow("rotated");
    namedWindow("notrotated");
    
    Mat target = originFrame(enlargedRect(originTrackingBox, trackRegionScale, false));
    Point2f displacement = enlargedRect(originTrackingBox, trackRegionScale, false).tl(); // displacement of tracking box
    Point2f center = Point2f(target.size().width / 2, target.size().height / 2); // new center
    
    vector<Rect> eyes, tempEyes;
    vector<Point2f> eyeCenters, tempEyeCenters;
    Mat rotMat, rotImg;
    double angle, angleSum(0), angleCount(0);
    for(int i=-10; i<10; i++){
        angle = tbAngle + i * step;
        if(angle < -60 || angle > 60)
            continue;
        // computing parameters
        rotMat = getRotationMatrix2D(center, angle, 1);
        // do warping
        warpAffine(target, rotImg, rotMat, target.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));
        tempEyes = detectEyeAndFace(rotImg, false);

        if(tempEyes.size() > 0){// if find an eye
            // for tbAngle
            angleSum += angle;
            angleCount++;
            // clean previous data
            tempEyeCenters.resize(0);
            // rotate the center back
            Point2f ptr, tempDis;
            double tempLength, tempAngle, ptrAngle;
            for(size_t j=0; j<tempEyes.size(); j++){
                ptr = Point2f(tempEyes[j].tl().x + tempEyes[j].width / 2, tempEyes[j].tl().y + tempEyes[j].height / 2);
                
                circle(rotImg, center, 1, Scalar(0, 255, 0), 3);
                line(rotImg, Point(target.cols, center.y), Point(0, center.y), Scalar(0, 255, 0));
                line(rotImg, Point(center.x, target.rows), Point(center.x, 0), Scalar(0, 255, 0));
                circle(target, center, 1, Scalar(0, 255, 0), 3);
                line(target, Point(target.cols, center.y), Point(0, center.y), Scalar(0, 255, 0));
                line(target, Point(center.x, target.rows), Point(center.x, 0), Scalar(0, 255, 0));
                circle(rotImg, ptr, 1, Scalar(255, 255, 0), 3);
                line(rotImg, ptr, center, Scalar(0, 255, 0));
                imshow("rotated", rotImg);
                
                tempDis = ptr - center;
                tempLength = sqrt(tempDis.x * tempDis.x + tempDis.y * tempDis.y);
                
                if(tempDis.x >= 0){ // shit about math!!!!!
                    tempDis.x += 0.1;
                    tempAngle = atan(tempDis.y / tempDis.x);
                }
                else{
                    tempDis.x -= 0.1;
                    tempAngle = atan(tempDis.y / tempDis.x) + CV_PI;
                }

                ptrAngle = tempAngle + angle * CV_PI / 180;
                ptr.x = tempLength * cos(ptrAngle);
                ptr.y = tempLength * sin(ptrAngle);
                ptr += center;
                
                circle(target, ptr, 1, Scalar(255, 255, 0), 3);
                line(target, ptr, center, Scalar(0, 255, 0));
                imshow("notrotated", target);
                
                ptr += displacement;
                tempEyeCenters.push_back(ptr);
            }
            eyes.insert(eyes.end(), tempEyes.begin(), tempEyes.end());
            eyeCenters.insert(eyeCenters.end(), tempEyeCenters.begin(), tempEyeCenters.end());
        }
    }
    
    if(eyes.size() == 0){
        if(trackRegionScale == 1){
            isLostFrame = true;
            lostFrame += 0.05;
            tuneByDetection(5, 2.5);
            return false;
        }
        else{
            isLostFrame = true;
            lostFrame++;
            return false;
        }
    }
    else{
        tbAngle = angleSum / angleCount;
        if(trackRegionScale == 1){
            isLostFrame = false;
            lostFrame = 0;
        }
    }
    
    Point2f averageCenter = Point2f(0, 0);
    double pointCount(0), averageSide(0);
    // using average to tune the percise eye position
    for(size_t i=0; i<eyes.size(); i++){
        if(sqrt(pow(eyeCenters[i].x - 2 * tbCenter.x, 2) + pow(eyeCenters[i].y - 2 * tbCenter.y, 2)) <= tbWidth * 2){
            averageSide += eyes[i].width;
            averageCenter += eyeCenters[i];
            pointCount++;
        }
    }
    if(pointCount > 0){
        averageCenter /= pointCount * 2;
        averageSide /= pointCount * 2;
        
        circle(target, averageCenter, 6, Scalar(0, 0, 255), 6);
        imshow("tune", target);
//        waitKey(0);

        tbCenter = averageCenter;
        tbWidth = (tbWidth / rectForTrackPercentage * (1 - tuningPercentage) + averageSide * tuningPercentage) * rectForTrackPercentage;
        tbHeight = tbWidth;
//        kMeansTuning(eyeCenters);
        getTrackingBox();
    }
    return true;
}

void EyeTracker::checkIsTracking(){
    if(lostFrame > maxLostFrame)
        is_tracking = false;
}

void EyeTracker::kMeansTuning(vector<Point2f> &eyeCenters){ // way too slow method…… though has significant effect
    namedWindow("kmeans");
    // doing clustering of detected eye centers
    Mat points(int(eyeCenters.size()), 1, CV_32FC2), labels;
    for(size_t r=0; r<eyeCenters.size(); r++){
        points.row(int(r)) = Scalar(eyeCenters[r].x, eyeCenters[r].y);
        circle(originFrame, eyeCenters[r], 3, Scalar(255, 255, 255), 3);
    }
    int clusterCount = min(int(eyeCenters.size()), 3);
    Mat centers(clusterCount, 1, points.type());
    kmeans(points, clusterCount, labels,
           TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 1, 5.0),
           1, KMEANS_PP_CENTERS, centers);
    eyeCenters.resize(clusterCount);
    for(size_t i=0; i<eyeCenters.size(); i++){
        eyeCenters[i] = centers.at<Point2f>(int(i));
        circle(originFrame, eyeCenters[i], 3, Scalar(0, 255, 0), 3);
    }
    imshow("kmeans", originFrame);
    // waitKey(0);
    
    // looking for the closest cluster
    double minDis(100), seq(0), temp;
    for(int i=0; i<clusterCount; i++){
        temp = sqrt(pow(eyeCenters[i].x - 2 * tbCenter.x, 2) + pow(eyeCenters[i].y - 2 * tbCenter.y, 2));
        if(temp < minDis){
            minDis = temp;
            seq = i;
        }
    }
    
    tbCenter = eyeCenters[seq] / 2;
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
        eyes_cascade.detectMultiScale( src, eyes, 1.1, 3, 0, Size(35, 35));
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

void EyeTracker::getTrackingBox(){
    trackingBox = enlargedRect(Rect(Point(tbCenter.x - tbWidth / 2, tbCenter.y - tbHeight / 2), Point(tbCenter.x + tbWidth / 2, tbCenter.y + tbHeight / 2)), 1);
    originTrackingBox = Rect(trackingBox.tl() / scale, trackingBox.br() / scale);
}

void EyeTracker::drawTrackingBox(Mat &dst){
    if(dst.empty())
        originFrame.copyTo(dst);
    
    getTrackingBox();
    if(!trackingBox.empty())
        rectangle(dst, originTrackingBox, Scalar(0, 0, 255), 3);
}

Rect EyeTracker::enlargedRect(Rect src, float times, bool isDefault){
    Point tl, br;
    Size size;
    if(isDefault)
        size = curFrame.size();
    else
        size = originFrame.size();
    
    tl.x = max(src.tl().x - src.width * (times - 1) / 2, 0);
    tl.x = min(tl.x, size.width);
    tl.y = max(src.tl().y - src.height * (times - 1) / 2, 0);
    tl.y = min(tl.y, size.height);
    br.x = min(src.br().x + src.width * (times - 1) / 2, size.width);
    br.x = max(br.x, 0);
    br.y = min(src.br().y + src.height * (times - 1) / 2, size.height);
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
    
    if(point[0].size() == 0)
        return;
    
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
// efficiency related:
void EyeTracker::setTimeStart(){
    start = clock();
}

void EyeTracker::setTimeEnd(){
//    sumTime += clock() - start;
//    timeCount++;
    sumTime = clock() - start;
}

double EyeTracker::getAverageTime(){
//    return double(sumTime) / (timeCount * CLOCKS_PER_SEC) * 1000;
    return sumTime / 1000;
}
