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
    // open outputfile
    stringstream ss;
    string buffer, fileName;
    time_t tt = time(NULL);
    tm* t= localtime(&tt);
    ss << "Date_" << t->tm_year + 1900 << "." << t->tm_mon + 1 << "." << t->tm_mday << "_Time_" << t->tm_hour << ":" << t->tm_min << ":" << t->tm_sec;
    ss >> buffer;
    fileName = buffer + ".csv";
    cout << fileName << endl;
    outputFile.open(fileName.c_str());
    outputFile << "Test infomation , " << buffer << endl;
    outputFile << "per minute: , " << ", " << "per ten minutes: , " << endl;

    //  init face and eye casacade
    if( !face_cascade.load( face_cascade_name ) ){
        printf("face_cascade_name加载失败\n");
        getchar();
    }
    if( !eyes_cascade.load( eyes_cascade_name ) ){
        printf("eye_cascade_name加载失败\n");
        getchar();
    }
    // init blink counter
    startMin = clock();
    blinkCounter = 0;
    timeCounter = 0;
}

EyeTracker::~EyeTracker(){
    outputFile.close();
}

int EyeTracker::finalProduct(){
    VideoCapture cap;
    Mat frame;
    
    cap.open(0);
    
    if(!cap.isOpened()){
        cout << "fail to open camera" << endl;
        return -1;
    }
    
    namedWindow("camera");
    
    while(waitKey(1) != 27){
        cap >> frame;
        frame.copyTo(originFrame);
        setTimeStart();
        trackByOptFlow(0.5);
        tuneByDetection(5, 0.5);
        isBlink = blinkDetection();
        setTimeEnd();
        drawTrackingBox(frame);
        writeBlinkToFile();
        imshow("camera", frame);
        cout << getAverageTime() << endl;
    }
    return 0;
}

void EyeTracker::writeBlinkToFile(){
    if(isBlink){
        blinkCounter++;
        blinkCounterTen++;
    }
    if(clock() - startMin >= 1 * CLOCKS_PER_SEC){
        outputFile << ++timeCounter << " , " << blinkCounter;
        startMin = clock();
        blinkCounter = 0;
        if(timeCounter % 10 == 0){
            outputFile << " , " << timeCounter / 10 << " , " << blinkCounterTen / 10;
            blinkCounterTen = 0;
        }
        else
            outputFile << " , , ";
        outputFile << endl;
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
                trackingBox = eyes[random()%eyes.size()];
                //recording the info of tracking box
                tbWidth = trackingBox.width;
                tbHeight = trackingBox.height;
                tbCenter = trackingBox.tl() + Point(tbWidth / 2, tbHeight / 2);
                getTrackingBox();
                // re_initiating variables
                optDisplacement = Point2f(0, 0);
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
                    optDisplacement = filteredDisplacement();
                    tbCenter += Point(optDisplacement);
                    getTrackingBox();
                }
            }
            else
                is_tracking = false;
        }
    }
}

bool EyeTracker::tuneByDetection(double step, double inputScale, double trackRegionScale){ // when eye movements from opt is small, we do not do tuning
    // decide if tuning is necessary or practical
    if((!is_tracking || getDis(optDisplacement) < 1) && !(badEyeCount >= 2)){
        isTuning = false;
        return false;
    }
    else
        isTuning = true;

    // rescaling
    Mat scaledFrame;
//    rescalePyr(originFrame, scaledFrame, inputScale);
    rescaleSize(originFrame, scaledFrame, inputScale);

    Rect scaledTrackingBox = Rect(originTrackingBox.tl() * inputScale, originTrackingBox.br() * inputScale);
    Mat target = scaledFrame(enlargedRect(scaledTrackingBox, trackRegionScale, inputScale));
    Point2f displacement = enlargedRect(scaledTrackingBox, trackRegionScale, inputScale).tl(); // displacement of tracking box
    Point2f center = Point2f(target.size().width / 2, target.size().height / 2); // new center
    
    vector<Rect> eyes, tempEyes;
    vector<DisFilter> dis;
    DisFilter tempDis;
    Mat rotMat, rotImg;
    double angle, angleSum(0), angleCount(0), averageAngleChange;
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
            lostFrame += 0.5;
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
        averageAngleChange = angleSum / angleCount - tbAngle;
        if(trackRegionScale == 1){
            isLostFrame = false;
            lostFrame = 0;
        }
    }
    
    double pointCount(0);
    // filt away the most far away centers:
    sort(dis.begin(), dis.end(), compDis);
    for(size_t i=0; i<(dis.size() > 4) ? (dis.size() * centerFilterPercentage) : 0; i++){ // do not do searching when there is too few eye detected
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

        // updating tracking box parameter
        // do para tuning:
        getOptDisTunedParameter();
        // we also have faith in optical flow and we will not abandon that!!!
        tbAngle += averageAngleChange * tuningPercentageForAngle;
        tbCenter += Point(averageCenterDisplacement) *  (inputScale == 1 ? tuningPercentageForCenter : tuningPercentageForCenterConst); // if it is for large scale tuning, believe in tuning
        // do some math to prevent over shrinking of tbWidth
        tbWidth = (tbWidth / rectForTrackPercentageConst * (1 - tuningPercentageForSide) + averageSide * tuningPercentageForSide) * rectForTrackPercentageConst;
        tbHeight = tbWidth;

        getTrackingBox();
    }
    
//    kMeansTuning(eyeCenters, inputScale);
//    getTrackingBox();
    return true;
}

void EyeTracker::getOptDisTunedParameter(){
    double para = 1 / (1 + exp(-getDis(optDisplacement) + 5));
    double paraForAngle = 1 / (1 + exp(-getDis(optDisplacement) + 3));
    tuningPercentageForSide = para * tuningPercentageForSideConst;
    tuningPercentageForCenter = para * tuningPercentageForCenterConst;
    tuningPercentageForAngle = paraForAngle * tuningPercentageForAngleConst;
}

void EyeTracker::checkIsTracking(){
    if(lostFrame >= maxLostFrame && is_tracking){
        is_tracking = false;
    }
}

bool EyeTracker::getEyeRegionWithCheck(){ // only return true when consecutive two confidently grabed eye is stored in cur/prevEye
    if(!is_tracking)
        return false;
    
    // count low quality grabed eyes: unmatched result between tune and optflow, no eye found
    bool isBadEye;
    if(isTuning)
        isBadEye = isLostFrame || getDis(averageCenterDisplacement) > tbWidth * 0.15 || getDis(optDisplacement) > 35;
    else{ // this is to assist checking if there is a eye grabed if tuning is not there
        vector<Rect> tempEyes;
        isLostFrame = !findMostRightEyes(detectEyeAtAngle(originFrame(enlargedRect(originTrackingBox, 1, 1)), tbAngle, Size(40, 40)), tempEyes);
        isBadEye = isLostFrame;
    }
    if(badEyeCount > 0 && !isBadEye) // when a good eye is found, re_init the badEyeCount
        badEyeCount = 0;
    badEyeCount += isBadEye;
    
    // declare fail to grab eye when two consecutive bad eye detected or no eye grabed at all
    if(enlargedRect(originTrackingBox, 1, 1).empty() || badEyeCount >= 2 ){
        if(!prevEye.mat.empty()){
            prevEye = Eye();
            curEye = Eye();
        }
        else if(!curEye.mat.empty())
            curEye = Eye();
        isDoubleCheck = false;
        isThisFrame = 0;
        return false;
    }
    
    // store data
    prevEye = curEye;
    // get curEye Mat
    originFrame(enlargedRect(originTrackingBox, 1, 1)).copyTo(curEye.mat);
    cvtColor(curEye.mat, curEye.mat, COLOR_BGR2GRAY);
    if(tbAngle != 0){  // do rotation
        Mat rotMat = getRotationMatrix2D(Point2f(curEye.mat.cols / 2, curEye.mat.rows / 2), tbAngle, 1); // get rotation matrix
        warpAffine(curEye.mat, curEye.mat, rotMat, curEye.mat.size(), INTER_LINEAR, BORDER_DEFAULT);
    }
    // get rect in curEye
    double widthPortion = 0.7, heightPortion = 0.5; // no larger than 3 !!!
    Point2f tempCenter = Point2f(tbOrgWidth / 2, tbOrgHeight / 2 + tbOrgHeight / 8);
    Rect smallEyeRegion = Rect(tempCenter - Point2f(tbOrgWidth * widthPortion, tbOrgHeight * heightPortion) / 2, tempCenter + Point2f(tbOrgWidth * widthPortion, tbOrgHeight * heightPortion) / 2);
    curEye.rect = enlargedRect(smallEyeRegion, 1, 0, true);
    // make the center of cureye closer to the center of iris
    Mat tempMat = curEye.getEye();
    if(tempMat.empty()){
        badEyeCount++;
        return false;
    }
    Point2f irisDisplacement;
    double bufferDev;
    curEye.thresh = getThresholdEstimation(tempMat);
    irisDisplacement = thresholdWithGrayIntegralFiltering(tempMat, tempMat, curEye.thresh, bufferDev, false);
    if(getDis(irisDisplacement) <= curEye.rect.width * 0.5)
        curEye.rect += Point(irisDisplacement) * 0.5;
    curEye.rect = enlargedRect(curEye.rect, 1, 0, true);
    // update curEye deviation
    tempMat = curEye.getEye();
    curEye.thresh = getThresholdEstimation(tempMat);
    thresholdWithGrayIntegralFiltering(tempMat, tempMat, curEye.thresh, curEye.dev);
    
    // wait until two eyes are grabed
    if(prevEye.mat.empty()){
        return false;
    }
    if(isDoubleCheck && isThisFrame++ < waitFrame)
        return false;
    return true;
}

bool EyeTracker::blinkDetection(){
    // check if data required for blink detection is ready
    if(!getEyeRegionWithCheck())
        return false;
    
    Mat tempOpenEye, tempClosedEye, tempPrevEye, tempCurEye;
    double openDevSameThresh, closeDevSameThresh, openDevIntrinsic, closeDevIntrinsic, sameChange, intriChange;
    bool isBlinkPossible = false;

    if(!isDoubleCheck){
        tempCurEye = curEye.getEye();
        openDevSameThresh = prevEye.dev;
        openDevIntrinsic = prevEye.dev;
        thresholdWithGrayIntegralFiltering(tempCurEye, tempClosedEye, prevEye.thresh, closeDevSameThresh);
        closeDevIntrinsic = curEye.dev;
    }
    else{
        tempPrevEye = assumedClosedEye.getEye();
        openDevSameThresh = curEye.dev;
        openDevIntrinsic = curEye.dev;
        thresholdWithGrayIntegralFiltering(tempPrevEye, tempClosedEye, curEye.thresh, closeDevSameThresh);
        closeDevIntrinsic = assumedClosedEye.dev;
    }
    
    sameChange = (openDevSameThresh - closeDevSameThresh) / openDevSameThresh;
    intriChange = (openDevIntrinsic - closeDevIntrinsic) / openDevSameThresh;
    sameChange = sameChange > 0 ? sameChange : 0;
    intriChange = intriChange > 0 ? intriChange : 0;
    cout << "same: " << sameChange << endl;
    cout << "intr: " << intriChange << endl;
    cout << "port: " << (sameChange + intriChange) / 2 << endl;
    if(!isDoubleCheck && (sameChange + intriChange) / 2 >= devChangeThreshold)
        isBlinkPossible = true;
    else if(isDoubleCheck && (sameChange + intriChange) / 2 >= devChangeThreshold * 0.8)
        isBlinkPossible = true;
    if(isBlinkPossible){ // estimate a probable blink
        if(isDoubleCheck){
            isDoubleCheck = false;
            isThisFrame = 0;
            return true;
        }
        else{
            assumedClosedEye = curEye;
            isDoubleCheck = true;
            isThisFrame = 0;
        }
    }
    else if(isDoubleCheck){
        isDoubleCheck = false;
        isThisFrame = 0;
    }

    return false;
}

int EyeTracker::getBlackPixNo(Mat src){
    uchar* ptr = src.ptr<uchar>(0);
    int counter(0);
    for(int i=0; i<src.rows * src.cols; i++)
        if(ptr[i] == 0)
            counter++;
    return counter;
}

Point2f EyeTracker::thresholdWithGrayIntegralFiltering(Mat &src, Mat &dst, double tempThreshold, double &dev, bool isGetDev){
    // do init threshold
    threshold(src, dst, tempThreshold, 255, THRESH_BINARY);
    medianBlur(dst, dst, 3);
    
    // do integral projection:
    Mat paintX = Mat::zeros( dst.rows, dst.cols, CV_8UC1 );
    Mat paintY = Mat::zeros( dst.rows, dst.cols, CV_8UC1 );
    int* v = new int[dst.cols];
    int* h = new int[dst.rows];
    uchar* myptr;
    int x,y, temp;
    // get cumulative projection
    // get vertical projection
    for( x=0; x<dst.cols; x++)
    {
        temp = 0;
        v[x] = x == 0 ? 0 : v[x-1];
        for(y=0; y<dst.rows; y++)
        {
            myptr = dst.ptr<uchar>(y);        //逐行扫描，返回每行的指针
            if( myptr[x] == 0 )
                temp++;
        }
        if(temp > dst.rows * 0.9){ // reject large shades
            for(y=0; y<src.rows; y++) // clear the rejected col
            {
                //myptr = src.ptr<uchar>(y);
                //myptr[x] = 255;
                myptr = dst.ptr<uchar>(y);
                myptr[x] = 255;
            }
            temp = 0;
        }
        v[x] += temp;
    }
    // draw vertical projection
    for( x=0; x<dst.cols; x++)
    {
        for(y=0; y < (x == 0 ? v[0] : (v[x] - v[x-1])); y++)
        {
            paintX.ptr<uchar>(y)[x] = 255;
        }
    }
    // get horizontal projection
    for( x=0; x<dst.rows; x++)
    {
        h[x] = x == 0 ? 0 : h[x-1];
        temp = 0;
        myptr = dst.ptr<uchar>(x);
        for(y=0; y<dst.cols; y++)
        {
            if( myptr[y] == 0 )
                temp++;
        }
        if(temp > dst.cols * 0.8){ //reject long bars
            //myptr = src.ptr<uchar>(x); //clear the row
            //for(y=0; y<dst.cols; y++){
            //    myptr[y] = 255;
            //}
            myptr = dst.ptr<uchar>(x); //clear the row
            for(y=0; y<dst.cols; y++){
                myptr[y] = 255;
            }
            temp = 0; // reject long bars
        }
        h[x] += temp;
    }
    // draw horizontal projection
    for( x=0; x<dst.rows; x++)
    {
        myptr = paintY.ptr<uchar>(x);
        for(y=0; y < (x == 0 ? h[0] : (h[x] - h[x-1])); y++)
        {
            myptr[y] = 255;
        }
    }

    // grab position of the iris
    vector<Point2f> H, V; // x is coordinate, y is result no.
    int irisWidth = curEye.rect.height / 3, extention = irisWidth * 0.1;
    int centralNo, lateralNo, resultNo;
    Point checkH = Point(0, irisWidth + 2 * extention), checkV = Point(0, irisWidth + 2 * extention), step = Point(2, 2);
    for(checkH.x = 0; checkH.y < dst.cols; checkH += step){
        lateralNo = v[checkH.x + extention] - v[checkH.x] + v[checkH.y] - v[checkH.y - extention];
        centralNo = v[checkH.y - extention] - v[checkH.x + extention];
        resultNo = centralNo - lateralNo;
        //if(resultNo >= irisWidth * irisWidth / 3)
        if(resultNo >= irisPixels - 10)
            H.push_back(Point((checkH.x + checkH.y) / 2, resultNo));
    }
    for(checkV.x = 0; checkV.y < dst.rows; checkV += step){
        lateralNo = h[checkV.x + extention] - h[checkV.x] + v[checkV.y] - v[checkV.y - extention];
        centralNo = h[checkV.y - extention] - h[checkV.x + extention];
        resultNo = centralNo - lateralNo;
        //if(resultNo >= irisWidth * irisWidth / 3)
        if(resultNo >= irisPixels - 10)
            V.push_back(Point((checkV.x + checkV.y) / 2, resultNo));
    }
    double dif, tempDif, sumPixels(0);
    Point2f tempCenter, sumCenter(Point2f(0, 0));
    int counter(0);
    for(int i=0; i<H.size(); i++){
        dif = 0.3;
        for(int j=0; j<V.size(); j++){
            tempDif = abs(H[i].y - V[j].y) / H[i].y;
            if(tempDif < dif){
                dif = tempDif;
                tempCenter = Point2f(H[i].x, V[j].x);
            }
        }
        if(dif < 0.3){
            sumPixels += H[i].y;
            sumCenter += tempCenter;
            counter++;
            circle(paintX, tempCenter, 3, 255, 3);
        }
    }
    sumPixels /= counter + 0.1;
    sumCenter /= counter + 0.1;
    if(sumCenter != Point2f(0, 0))
        sumCenter -= Point2f(src.cols/2, src.rows/2);
    irisPixels = 0.5 * irisPixels+ 0.5 * (0.5 * sumPixels + 0.5 * irisWidth * irisWidth);
    
    if(isGetDev){
        // calculate dev for blink detection use
        double devX, meanX, devY, meanY;
        double targetRegionLength = irisWidth * 2;
        int regionStart, actualLen;
        Point2f newCenter = 0.3 * sumCenter + Point2f(src.cols/2, src.rows/2);
        // turn accumulative to non accumulative
        for(int i=dst.cols-1; i>=0; i--){
            if(i != 0){
                v[i] = v[i] - v[i-1];
            }
        }
        regionStart = (newCenter.x - targetRegionLength / 2) >= 0 ? (newCenter.x - targetRegionLength / 2) : 0;
        actualLen = (regionStart + targetRegionLength) < dst.cols ? targetRegionLength : (dst.cols - 1);
        devX = getDev(v + regionStart, actualLen, meanX);
        for(int i=dst.rows-1; i>=0; i--){
            if(i != 0){
                h[i] = h[i] - h[i-1];
            }
        }
        regionStart = (newCenter.y - targetRegionLength / 2) >= 0 ? (newCenter.y - targetRegionLength / 2) : 0;
        actualLen = (regionStart + targetRegionLength) < dst.rows ? targetRegionLength : (dst.rows - 1);
        devY = getDev(h + regionStart, actualLen, meanY);
        
        dev = devX;
        
        // for good looking data analysis:
        stringstream s;
        string X, Y, tempDev, tempMean;
        
        s << int(devX);
        s >> tempDev;
        s << meanX;
        s >> tempMean;
        X = tempDev;
        putText(paintX, X, Point(0, paintX.cols / 2), FONT_HERSHEY_PLAIN, 1, 255);
    }
    namedWindow("dev");
    imshow("dev", paintX);
    namedWindow("thr");
    imshow("thr", dst);
    return sumCenter;
};

double EyeTracker::getThresholdEstimation(Mat &src){ // do gray integral to threshold image
    // blurred eyes
    Mat blurEye;
    medianBlur(src, blurEye, 3);
    
    // define the threshold by checking the lines of pixels around the center
    Point tempCenter = Point2f(blurEye.cols / 2 - 2, blurEye.rows / 2 - 2);
    Point iter;
    double divideHor = 3, divideVer = 4;
    int hor = blurEye.cols / divideHor, ver = blurEye.rows / divideVer;
    
    uchar* grayH, *grayV;
    double minH(0), minV(0);
    grayH = new uchar[2*hor];
    grayV = new uchar[2*ver];
    // get appropriate threshold: iterate for three pair of crosses.
    double tempThreshold(0);
    for(int i=0; i<3; i++){
        tempCenter.x += 2 * i;
        tempCenter.y += 2 * i;
        iter.x = tempCenter.x;
        for(iter.y = tempCenter.y - ver; iter.y < tempCenter.y + ver; iter.y++){ // iterate horizontal
            grayV[iter.y - (tempCenter.y - ver)] = blurEye.at<uchar>(iter);
        }
        sort(grayV, grayV + 2*ver);
        for(int i=0; i<int(2*ver * thresholdFilterPercentage); i++){
            minH += grayV[i];
        }
        minH /= int(2 * ver * thresholdFilterPercentage) + 0.001;
        iter.y = tempCenter.y;
        for(iter.x = tempCenter.x - hor; iter.x < tempCenter.x + hor; iter.x++){// iterate vertical
            grayH[iter.x - (tempCenter.x - hor)] = blurEye.at<uchar>(iter);
        }
        sort(grayH, grayH + 2*hor);
        for(int i=0; i<int(2*hor * thresholdFilterPercentage); i++){
            minV += grayH[i];
        }
        minV /= int(2 * hor * thresholdFilterPercentage) + 0.001;
        tempThreshold += (minV * divideHor + minH * divideVer) / (divideVer + divideHor) + 2;
    }
    
    tempThreshold /= 3;
    
    Mat temp;
    double blackPortion;
    for(int i=0; i<10; i++){
        threshold(blurEye, temp, tempThreshold, 255, THRESH_BINARY);
        blackPortion = getBlackPixNo(temp) / double(temp.rows * temp.cols);
        if(blackPortion > 0.18 || getBlackPixNo(temp) > irisPixels * 1.5)
            tempThreshold -= 2;
        else if(blackPortion < 0.08 || getBlackPixNo(temp) < irisPixels * 0.25)
            tempThreshold += 2;
        else
            return tempThreshold;
    }
    return tempThreshold;
}

Point2f EyeTracker::opticalFlowForBlinkDetection(){ // useless for now
    vector<Point2f> eyePoints[2];
    goodFeaturesToTrack(prevEye.getEye(), eyePoints[0], 50, 0.01, 5);
    
    if(eyePoints[0].size() == 0)
        return Point2f(0, 0);
    
    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(prevEye.getEye(), curEye.getEye(), eyePoints[0], eyePoints[1], status, err);
    double counter(0);
    for(int i=0; i<eyePoints[0].size(); i++)
        if(status[i])
            counter++;
    
    int k(0);
    DisFilter tempDis;
    vector<DisFilter> dis;
    // store valid feature points
    for (size_t i = 0; i<eyePoints[1].size(); i++)
    {
        double distance = getDis(eyePoints[0][i], eyePoints[1][i]);
        if (status[i] && distance > 2) // check if the point is qualified
        {
            eyePoints[1][k] = eyePoints[1][i];
            eyePoints[0][k] = eyePoints[0][i];
            // store the displacement of points for further filtering
            tempDis.dis = eyePoints[0][i] - eyePoints[1][i];
            tempDis.seq = k++;
            tempDis.flag = false;
            dis.push_back(tempDis);
        }
    }

    eyePoints[1].resize(k);
    eyePoints[0].resize(k);
    // sort displacement in descending order and do filtering
    size_t size = eyePoints[1].size();
    if(size - 2*size*blinkOptFilterPercentage > 5){ // do filtering when there are too few features
        vector<DisFilter>::iterator iter;
        sort(dis.begin(), dis.end(), compX);
        iter = dis.begin();
        for(int i=0; i<size*blinkOptFilterPercentage; i++){
            (iter++)->flag = true;
        }
        iter = dis.end();
        for(int i=0; i<size*blinkOptFilterPercentage; i++){
            (--iter)->flag = true;
        }
        
        sort(dis.begin(), dis.end(), compY);
        iter = dis.begin();
        for(int i=0; i<double(size*blinkOptFilterPercentage); i++){
            (iter++)->flag = true;
        }
        iter = dis.end();
        for(int i=0; i<double(size*blinkOptFilterPercentage); i++){
            (--iter)->flag = true;
        }
        
        for(iter=dis.begin(); iter != dis.end();){
            if(iter->flag == true){
                *(eyePoints[0].begin() + iter->seq) = Point2f(-101, -101);
                *(eyePoints[1].begin() + iter->seq) = Point2f(-101, -101);
                dis.erase(iter);
            }
            else
                iter++;
        }
        
        for(size_t i =0; i<eyePoints[0].size();){
            if(eyePoints[0][i] == Point2f(-101, -101)){
                eyePoints[0].erase(eyePoints[0].begin() + i);
                eyePoints[1].erase(eyePoints[1].begin() + i);
            }
            else
                i++;
        }
    }
    
    Mat dst = curEye.mat.clone();
    
    double disX(0), disY(0);
    for(size_t i=0; i<dis.size(); i++){
        disX += dis[i].dis.x;
        disY += dis[i].dis.y;
    }
    disX /= dis.size();
    disY /= dis.size();
    
    if(disX != disX){
        disX = 0;
        disY = 0;
    }
    Point2f tempCenter = Point2f(dst.cols / 2, dst.rows / 2);
    Point2f y = Point2f(tempCenter.x, tempCenter.y - disY), x = Point2f(tempCenter.x - disX, tempCenter.y );
    
    line(dst, tempCenter, y, Scalar(255), 3);
    line(dst, tempCenter, x, Scalar(255), 3);
    
    for (size_t i = 0; i<eyePoints[1].size(); i++)
    {
        line(dst, eyePoints[0][i], eyePoints[1][i], Scalar(0));
        circle(dst, eyePoints[1][i], 3, Scalar(0), -1);
    }
    
    return Point2f(0, 0);
}

bool EyeTracker::compDis(const DisFilter a, const DisFilter b){
    return pow(a.dis.x, 2) + pow(a.dis.y, 2) > pow(b.dis.x, 2) + pow(b.dis.y, 2);
}

void EyeTracker::kMeansTuning(vector<Point2f> &eyeCenters, double inputScale){ // way too slow method…… though has significant effect
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
    
    return tempScale;
}

double EyeTracker::rescaleSize(Mat src, Mat &dst, double inputScale){
    resize(src, dst, Size(0, 0), inputScale, inputScale);
    
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
    if(src.empty())
        return vector<Rect>();
    
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
    if(tbWidth / scale > 200)
        tbWidth = 200 * scale;
    else if(tbWidth / scale < 80)
        tbWidth = 80 * scale;
    trackingBox = enlargedRect(Rect(Point(tbCenter.x - tbWidth / 2, tbCenter.y - tbHeight / 2), Point(tbCenter.x + tbWidth / 2, tbCenter.y + tbHeight / 2)), 1, scale);
    originTrackingBox = Rect(trackingBox.tl() / scale, trackingBox.br() / scale);
    tbOrgWidth = tbWidth / scale;
    tbOrgHeight = tbHeight / scale;
    tbOrgCenter = tbCenter / scale;
}

void EyeTracker::drawTrackingBox(Mat &dst){
    if(dst.empty())
        originFrame.copyTo(dst);
    
    getTrackingBox();
    if(!trackingBox.empty()){
        if(isBlink)
            putText(dst, "Blink", originTrackingBox.tl() + Point(0, -3), FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255), 2, 8, 0);
        rectangle(dst, originTrackingBox, Scalar(0, 0, 255), 3);
    }
}

Rect EyeTracker::enlargedRect(Rect src, float times, double inputScale, bool isWithinTB){
    Point tl, br;
    Size size;
    if(isWithinTB)
        size = originTrackingBox.size();
    else
        size = Size(originFrame.size().width * inputScale, originFrame.size().height * inputScale);
    
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
    
    vector<uchar> status; // 跟踪特征的状态，特征的流发现为1，否则为0
    vector<float> err;
    
    calcOpticalFlowPyrLK(prevFrame(src), curFrame(src), tempPoint[0], tempPoint[1], status, err);
    point[1].resize(tempPoint[1].size());
    
    int k = 0;
    for (size_t i = 0; i<tempPoint[1].size(); i++)
    {
        if (status[i] && getDis(tempPoint[0][i], tempPoint[1][i]) > 0.1) // check if the point is qualified
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
    
    if(size - 2*size*filterPercentage - 10 > 0){ // if there is too little points, do not do filtering
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
        for(int i=0; i<double(size*filterPercentage); i++){
            (iter++)->flag = true;
        }
        iter = tempDis.end();
        for(int i=0; i<double(size*filterPercentage); i++){
            (--iter)->flag = true;
        }
        
        for(iter=tempDis.begin(); iter != tempDis.end();){
            if(iter->flag == true){
                *(point[0].begin() + iter->seq) = Point2f(-101, -101);
                *(point[1].begin() + iter->seq) = Point2f(-101, -101);
                tempDis.erase(iter);
            }
            else
                iter++;
        }
        
        for(size_t i =0; i<point[0].size();){
            if(point[0][i] == Point2f(-101, -101)){
                point[0].erase(point[0].begin() + i);
                point[1].erase(point[1].begin() + i);
                initPoint.erase(initPoint.begin() + i);
            }
            else
                i++;
        }
    }
    
    for(size_t i=0; i<tempDis.size(); i++){
        disX += tempDis[i].dis.x;
        disY += tempDis[i].dis.y;
    }
    disX /= tempDis.size();
    disY /= tempDis.size();
    
    if(disX != disX){
        disX = 0;
        disY = 0;
    }
    
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
