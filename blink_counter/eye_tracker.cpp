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

void EyeTracker::trackByOptFlow(double inputScale){
    if(curFrame.empty()){
        originFrame.copyTo(curFrame);
        inputScale = rescalePyr(curFrame, curFrame, inputScale);
        prevFrame.copyTo(colorFrame);
        cvtColor(curFrame, curFrame, COLOR_BGR2GRAY);
        scale = inputScale;
    }
    else{
        vector<Rect> eyes;

        // double start = getTickCount(); //to count the period the process takes

        //swap prev and cur
        curFrame.copyTo(prevFrame);
        swap(point[1], point[0]);
        //get current frame
        originFrame.copyTo(colorFrame);
        inputScale = rescalePyr(colorFrame, colorFrame, inputScale);
        colorFrame.copyTo(curFrame);
        cvtColor(curFrame, curFrame, COLOR_BGR2GRAY);
        
        // check if the scaling of prev frame is identical
        if(scale != inputScale){
            rescalePyr(prevFrame, prevFrame, inputScale / scale);
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
            double angle;
            vector<Rect> tempEyes;
            for(int i=0; i<20; i++){
                angle = tbAngle + int(i / 2) * pow(-1, i % 2) * 10;
                if(angle < -90 || angle > 90)
                    continue;
                // detect eys in rotated image
                tempEyes = detectEyeAtAngle(curFrame, angle, Size(30 * inputScale, 30 * inputScale));
                if(tempEyes.size() > 0){
                    tbAngle = angle;
                    break;
                }
            }
            if(findMostRightEyes(tempEyes, eyes)){
                trackingBox = eyes[0];
                //recording the info of tracking box
                tbWidth = trackingBox.width;
                tbHeight = trackingBox.height;
                tbCenter = trackingBox.tl() + Point(tbWidth / 2, tbHeight / 2);
                getTrackingBox();
                is_tracking = true;
                lostFrame = 0;
                isLostFrame = false;
            }
            else{
                tbAngle = 0;
                tbWidth = 0;
                tbHeight = 0;
                tbCenter = Point2f(0, 0);
                getTrackingBox();
                point[0].resize(0);
                point[1].resize(0);
                initPoint.resize(0);
            }
        }
        else{
            if(!enlargedRect(trackingBox, 2.5, scale).empty()){
                opticalFlow(enlargedRect(trackingBox, 2.5, scale));
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

bool EyeTracker::tuneByDetection(double step, double inputScale, double trackRegionScale){
    Rect scaledTrackingBox = Rect(originTrackingBox.tl() * inputScale, originTrackingBox.br() * inputScale);
    if(enlargedRect(scaledTrackingBox, trackRegionScale, inputScale).empty())
        return false;
    
    namedWindow("tune");
    // rescaling
    Mat scaledFrame;
//    rescalePyr(originFrame, scaledFrame, inputScale);
    rescaleSize(originFrame, scaledFrame, inputScale);

    Mat target = scaledFrame(enlargedRect(scaledTrackingBox, trackRegionScale, inputScale));
    Point2f displacement = enlargedRect(scaledTrackingBox, trackRegionScale, inputScale).tl(); // displacement of tracking box
    Point2f center = Point2f(target.size().width / 2, target.size().height / 2); // new center
    
    vector<Rect> eyes, tempEyes;
    vector<DisFilter> dis;
    DisFilter tempDis;
    Mat rotMat, rotImg;
    double angle, angleSum(0), angleCount(0);
    bool stopFlag[2]{false, false}; // signal the failure of finding an eye at 3 consecutive angles
    bool isFoundEye[2]{false, false}; // this is to make sure that at least one eye is found
    char sigFlag[2]{0, 0};
    for(int i=0; i<20; i++){
        if(!stopFlag[i%2]){ // check if cannot find eyes no more
            angle = tbAngle + int(i / 2) * pow(-1, i % 2) * 10;
            if(angle < -90 || angle > 90)
                stopFlag[i%2] = true;
        }
        else
            continue;
        // detect eys in rotated image
        if(findMostRightEyes(detectEyeAtAngle(target, angle, Size(40 * inputScale, 40 * inputScale)), tempEyes)){// if find an eye
            sigFlag[i%2] = 0;
            isFoundEye[i%2] = true;
            // for tbAngle
            angleSum += angle * tempEyes.size();
            angleCount += tempEyes.size();
            // record eye centers
            Point2f ptr;
            for(size_t i=0; i<tempEyes.size(); i++){
                ptr = Point2f(tempEyes[i].tl().x + tempEyes[i].width / 2, tempEyes[i].tl().y + tempEyes[i].height / 2);
                
                circle(target, ptr, 1, Scalar(255, 255, 0), 3);
                line(target, ptr, center, Scalar(0, 255, 0));

                ptr += displacement;
                ptr *= scale / inputScale;
                
                tempDis.dis = ptr - Point2f(tbCenter);
                tempDis.seq = angleCount - 1;
                tempDis.flag = false;
                
                dis.push_back(tempDis);
                eyes.push_back(tempEyes[i]);
            }
        }
        else if(sigFlag[i%2] == 3 && isFoundEye[i%2]) // only start to decide whether to stop searching when at least one eye is found
            stopFlag[i%2] = true;
        else if(sigFlag[i%2] < 3 && isFoundEye[i%2])
            sigFlag[i%2]++;
    }
    // check if search for the eye in a larger scale
    if(eyes.size() == 0){
        if(trackRegionScale == 1){
            lostFrame += 0.1;
            tuneByDetection(5, inputScale / 2, 3); // 2.5 can be slightly faster but play worse in tracking
            return false;
        }
        else{
            averageCenterDisplacement= Point2f(-1, -1);
            isLostFrame = true;
            lostFrame += 1;
            return false;
        }
    }
    else{
        tbAngle = angleSum / angleCount;
        isLostFrame = false;
        if(trackRegionScale == 1){
            lostFrame = 0;
        }
    }
    
    double pointCount(0);
    // filt away the most far away centers:
    sort(dis.begin(), dis.end(), compDis);
    for(size_t i=0; i<(dis.size() >= 2) ? (dis.size() * centerFilterPercentage) : 0; i++){ // do not do searching when there is too few eye detected
        eyes[dis[int(i)].seq] = Rect(0, 0, 0, 0);
        dis[i].dis = Point2f(0, 0);
    }
    // using average to tune the percise eye position
    averageCenterDisplacement= Point2f(0, 0);
    double averageSide(0);
    for(size_t i=0; i<eyes.size(); i++){
        if(eyes[i].width != 0){
            averageSide += eyes[i].width * scale / inputScale;
            averageCenterDisplacement += dis[i].dis;
            pointCount++;
        }
    }
    if(pointCount > 0){

        averageCenterDisplacement /= pointCount;
        averageSide /= pointCount;
        
        circle(target, (Point2f(tbCenter) + averageCenterDisplacement) / scale * inputScale - displacement, 2, Scalar(0, 0, 255), 2);
        rescaleSize(target, target, 1 / inputScale);
        imshow("tune", target);

        // updating tracking box parameter
        // we also have faith in optical flow and we will not abandon that!!!
        tbCenter += Point(averageCenterDisplacement) * tuningPercentageForCenter;
        // do some math to prevent over shrinking of tbWidth
        tbWidth = (tbWidth / rectForTrackPercentage * (1 - tuningPercentageForSide) + averageSide * tuningPercentageForSide) * rectForTrackPercentage;
        tbHeight = tbWidth;

        getTrackingBox();
    }
    
//    kMeansTuning(eyeCenters, inputScale);
//    getTrackingBox();
    return true;
}

void EyeTracker::checkIsTracking(){
    if(lostFrame > maxLostFrame){
        is_tracking = false;
    }
}

bool EyeTracker::getEyeRegionWithCheck(){ // only return true when consecutive two confidently grabed eye is stored in cur/prevEye
    // count low quality grabed eyes: unmatched result between tune and optflow, no eye found
    char isBadEye = isLostFrame || getDis(averageCenterDisplacement) > tbWidth * 0.15;
    badEyeCount += isBadEye;
    // declare fail to grab eye when two consecutive bad eye detected or no eye grabed at all
    if(enlargedRect(originTrackingBox, 1, 1).empty() || badEyeCount >= 2 ){
        if(!prevEye.empty()){
            prevEye = Mat();
            curEye = Mat();
        }
        else if(!curEye.empty())
            curEye = Mat();
        badEyeCount = 0;
        return false;
    }
    
    namedWindow("eye");
    
    prevEye = curEye.clone();
    // get curEye
    originFrame(enlargedRect(originTrackingBox, 1, 1)).copyTo(curEye);
    if(tbAngle != 0){
        Mat rotMat = getRotationMatrix2D(Point2f(curEye.cols / 2, curEye.rows / 2), tbAngle, 1); // get rotation matrix
        warpAffine(curEye, curEye, rotMat, curEye.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));    // rotate the image
    }
    imshow("eye", curEye);
    // wait until two eyes are grabed
    if(prevEye.empty())
        return false;
    return true;
}

bool EyeTracker::blinkDetection(){
    if(!getEyeRegionWithCheck())
        return false;
    // check if there is a tracked eye (need a check function!!!!)
    
    namedWindow("residue");
    resize(prevEye, prevEye, curEye.size());
    Mat residue = curEye - prevEye;
    
    imshow("residue", residue);
    
    prevEye = curEye.clone();
    return true;
}

bool EyeTracker::compDis(const DisFilter a, const DisFilter b){
    return pow(a.dis.x, 2) + pow(a.dis.y, 2) > pow(b.dis.x, 2) + pow(b.dis.y, 2);
}

void EyeTracker::kMeansTuning(vector<Point2f> &eyeCenters, double inputScale){ // way too slow method…… though has significant effect
    namedWindow("kmeans");
    // doing clustering of detected eye centers
    Mat points(int(eyeCenters.size()), 1, CV_32FC2), labels;
    for(size_t r=0; r<eyeCenters.size(); r++){
        points.row(int(r)) = Scalar(eyeCenters[r].x, eyeCenters[r].y);
        circle(originFrame, eyeCenters[r] / scale, 3, Scalar(255, 255, 255), 3);
    }
    int clusterCount = min(int(eyeCenters.size()), 3);
    Mat centers(clusterCount, 1, points.type());
    kmeans(points, clusterCount, labels,
           TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 1, 5.0),
           1, KMEANS_PP_CENTERS, centers);
    eyeCenters.resize(clusterCount);
    for(size_t i=0; i<eyeCenters.size(); i++){
        eyeCenters[i] = centers.at<Point2f>(int(i));
        circle(originFrame, eyeCenters[i] / scale, 3, Scalar(0, 255, 0), 3);
    }
    imshow("kmeans", originFrame);
    // waitKey(0);
    
    // looking for the closest cluster
    double minDis(100), seq(0), temp;
    for(int i=0; i<clusterCount; i++){
        temp = sqrt(pow(eyeCenters[i].x - tbCenter.x, 2) + pow(eyeCenters[i].y - tbCenter.y, 2));
        if(temp < minDis){
            minDis = temp;
            seq = i;
        }
    }
    
    tbCenter = eyeCenters[seq];
}

double EyeTracker::rescalePyr(Mat src, Mat &dst, double inputScale){
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
        cout << "Pyr:rescaling successful" << endl;
    
    return tempScale;
}

double EyeTracker::rescaleSize(Mat src, Mat &dst, double inputScale){
    resize(src, dst, Size(0, 0), inputScale, inputScale);
    cout << "Size:rescaling successful" << endl;
    
    return inputScale;
}

vector<Rect> EyeTracker::detectEyeAndFace(Mat src, Size minEye, bool isFace){
    vector<Rect> eyes;
    if(isFace){
        vector<Rect> faces;
        
        face_cascade.detectMultiScale( src, faces, 1.1, 2, 0, Size(50, 50));
        
        for( size_t i = 0; i < faces.size(); i++)
        {
            Mat faceROI = src( faces[i] );
            
            eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0, minEye);
            
            for( size_t j = 0; j < eyes.size(); j++)
                eyes[j] += faces[i].tl();
        }
    }
    else{
        eyes_cascade.detectMultiScale( src, eyes, 1.1, 2, 0, minEye);
    }
    
    return eyes;
}

bool EyeTracker::findMostRightEyes(vector<Rect> eyes, vector<Rect> &rightEyes){ // get all the eyes that is righter
    if(eyes.size() == 0)
        return false;
    
    rightEyes.assign(eyes.begin(), eyes.end());
    float averageX(0);
    for(size_t i=0; i<eyes.size(); i++)
        averageX += rightEyes[i].tl().x + rightEyes[i].width / 2;
    averageX /= rightEyes.size();
    // clear rects on the left
    for(vector<Rect>::iterator iter=rightEyes.begin(); iter != rightEyes.end();){
        if(iter->tl().x + iter->width / 2 < averageX - tbWidth / 8)
            rightEyes.erase(iter);
        else
            iter++;
    }
    
    return true;
}

vector<Rect> EyeTracker::detectEyeAtAngle(Mat src, double angle, Size minEye, bool isFace){
    Mat rotImg;
    Point2f center = Point2f(src.size().width / 2, src.size().height / 2), eyeCenter, tl, br; // new center
    if(angle != 0){
        Mat rotMat = getRotationMatrix2D(center, angle, 1); // get rotation matrix
        warpAffine(src, rotImg, rotMat, src.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));    // rotate the image
    }
    else
        src.copyTo(rotImg);
    vector<Rect> tempEyes = detectEyeAndFace(rotImg, minEye, isFace); // detect eyes
    
    for(size_t i=0; i<tempEyes.size(); i++){
        eyeCenter = Point2f(tempEyes[i].tl()) + Point2f(tempEyes[i].width / 2, tempEyes[i].height / 2);
        eyeCenter = rotatePoint(center, -angle, eyeCenter);
        tl = eyeCenter - Point2f(tempEyes[i].width / 2, tempEyes[i].height / 2);
        br = eyeCenter + Point2f(tempEyes[i].width / 2, tempEyes[i].height / 2);
    }
    return tempEyes;
}

Point2f EyeTracker::rotatePoint(Point2f center, double angle, Point2f ptr){
    if(angle != 0){
        Point2f tempDis;
        double tempLength, tempAngle, ptrAngle;
        
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
    }
    
    return ptr;
}

void EyeTracker::getTrackingBox(){
    trackingBox = enlargedRect(Rect(Point(tbCenter.x - tbWidth / 2, tbCenter.y - tbHeight / 2), Point(tbCenter.x + tbWidth / 2, tbCenter.y + tbHeight / 2)), 1, scale);
    originTrackingBox = Rect(trackingBox.tl() / scale, trackingBox.br() / scale);
}

void EyeTracker::drawTrackingBox(Mat &dst){
    if(dst.empty())
        originFrame.copyTo(dst);
    
    getTrackingBox();
    if(!trackingBox.empty())
        rectangle(dst, originTrackingBox, Scalar(0, 0, 255), 3);
}

Rect EyeTracker::enlargedRect(Rect src, float times, double inputScale){
    Point tl, br;
    Size size = Size(originFrame.size().width * inputScale, originFrame.size().height * inputScale);
    
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
