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
        
        //cout << "instant time: " << ((getTickCount() - start) / getTickFrequency()) * 1000 << endl;
        //sumTime += ((getTickCount() - start) / getTickFrequency()) * 1000; //output the time the process takes
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
    
    namedWindow("tune");
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
    
    cout << "                                        " <<optDisplacement << endl;
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
        tbAngle = angleSum / angleCount;
        isLostFrame = false;
        if(trackRegionScale == 1){
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
        imshow("tune", target);

        // updating tracking box parameter
        // do para tuning:
        getOptDisTunedParameter();
        // we also have faith in optical flow and we will not abandon that!!!
        tbCenter += Point(averageCenterDisplacement) * tuningPercentageForCenter;
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
    cout << "logistic para = " << para << endl;
    tuningPercentageForSide = para * tuningPercentageForSideConst;
    tuningPercentageForCenter = para * tuningPercentageForCenterConst;
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
        isBadEye = isLostFrame || getDis(averageCenterDisplacement) > tbWidth * 0.15 || getDis(optDisplacement) > 15;
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
        if(!prevEye.empty()){
            prevEye = Mat();
            curEye = Mat();
        }
        else if(!curEye.empty())
            curEye = Mat();
        return false;
    }
    
    namedWindow("eye");
    
    prevEye = curEye.clone();
    // get curEye
    cout << enlargedRect(originTrackingBox, 1, 1).empty() << endl;
    originFrame(enlargedRect(originTrackingBox, 1, 1)).copyTo(curEye);
    cvtColor(curEye, curEye, COLOR_BGR2GRAY);
    // do rotation
    if(tbAngle != 0){
        Mat rotMat = getRotationMatrix2D(Point2f(curEye.cols / 2, curEye.rows / 2), tbAngle, 1); // get rotation matrix
        //warpAffine(curEye, curEye, rotMat, curEye.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255)); // rotate the image
        warpAffine(curEye, curEye, rotMat, curEye.size(), INTER_LINEAR, BORDER_DEFAULT);
    }
    // cut out the middle region:
    double cutPortion = 2.75; // no larger than 3 !!!
    Point2f tempCenter = Point2f(tbOrgWidth / 2, tbOrgWidth / 2), downDisplacement = Point2f(0, tbOrgHeight / 8);
    Rect smallEyeRegion = Rect(tempCenter / cutPortion + downDisplacement, 2 * tempCenter - tempCenter / cutPortion + downDisplacement);
    // make sure the down displacement won't break out of the eye image...
    if(smallEyeRegion.br().y > curEye.rows)
        smallEyeRegion.height = curEye.rows - smallEyeRegion.y;
    // get the smaller eye
    curEye = curEye(smallEyeRegion);
    //normalizing and denoising
    //medianBlur(curEye, curEye, 3);
    //equalizeHist(curEye, curEye);
    resize(curEye, curEye, eyeSize);
    
    imshow("eye", curEye);
    
    // wait until two eyes are grabed
    if(prevEye.empty())
        return false;
    resize(prevEye, prevEye, curEye.size());
    return true;
}

bool EyeTracker::blinkDetection(){
    if(!getEyeRegionWithCheck())
        return false;
    // check if there is a tracked eye (need a check function!!!!)
    
    namedWindow("residue");
    // Mat residue = curEye - prevEye;
    Mat residue = curEye.clone();
    double averageGrayScale = 0;
    uchar* imagePtr = residue.ptr<uchar>(0);
    for(int i=0; i<residue.rows*residue.cols; i++){
        averageGrayScale += imagePtr[i];
    }
    averageGrayScale /= residue.rows*residue.cols;
    threshold(residue, residue, averageGrayScale / 2, 255, cv::THRESH_BINARY);
    // getHistogram();
    opticalFlowForBlinkDetection();
    //grayIntegral(residue, residue);
    //Mat ele = getStructuringElement(MORPH_RECT, Size(10, 10));
    //morphologyEx(residue, residue, MORPH_CLOSE, ele);
    imshow("residue", residue);

    return true;
}

void EyeTracker::getHistogram(){
    //图片数量nimages
    int nimages = 1;
    //通道数量,我们总是习惯用数组来表示，后面会讲原因
    int channels[1] = { 0 };
    //输出直方图
    Mat outputHist;
    //维数
    int dims = 1;
    //存放每个维度直方图尺寸（bin数量）的数组histSize
    int histSize[1] = { 256 };
    //每一维数值的取值范围ranges
    float hranges[2] = { 0, 255 };
    //值范围的指针
    const float *ranges[1] = { hranges };
    //是否均匀
    bool uni = true;
    //是否累积
    bool accum = false;
    
    //计算图像的直方图
    cv::calcHist(&curEye, nimages, channels, cv::Mat(), outputHist, dims, histSize, ranges, uni, accum);
    
    //找到最大值和最小值
    double maxValue = 0;
    double minValue = 0;
    cv::minMaxLoc(outputHist, &minValue, &maxValue, NULL, NULL);
    
    int height = 400;
    Mat histPic(height, histSize[0], CV_8U, cv::Scalar(255));
    
    // double rate = (histSize[0] / maxValue)*0.9;
    
    for (int i = 0; i < histSize[0]; i++)
    {
        //得到每个i和箱子的值
        float value = outputHist.at<float>(i);
        //画直线
        cv::line(histPic, cv::Point(i, height), cv::Point(i, height - value > 0 ? height - value : 0), Scalar(0));
    }
    namedWindow("hist");
    imshow("hist", histPic);
    return;
}

Point2f EyeTracker::opticalFlowForBlinkDetection(){
    vector<Point2f> eyePoints[2];
    goodFeaturesToTrack(prevEye, eyePoints[0], 50, 0.01, 5);
    
    if(eyePoints[0].size() == 0)
        return Point2f(0, 0);
    
    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(prevEye, curEye, eyePoints[0], eyePoints[1], status, err);
    
    int k(0);
    for (size_t i = 0; i<eyePoints[1].size(); i++)
    {
        if (status[i] && getDis(eyePoints[0][i], eyePoints[1][i]) > 0) // check if the point is qualified
        {
            eyePoints[1][k] = eyePoints[1][i];
            eyePoints[0][k++] = eyePoints[0][i];
        }
    }
    eyePoints[1].resize(k);
    eyePoints[0].resize(k);
    
    Mat dst = curEye.clone();
    for (size_t i = 0; i<eyePoints[1].size(); i++)
    {
        line(dst, eyePoints[0][i], eyePoints[1][i], Scalar(0, 0, 255));
        circle(dst, eyePoints[1][i], 3, Scalar(0, 255, 0), -1);
    }
    namedWindow("eyess");
    imshow("eyess", dst);
    
    return Point2f(0, 0);
}

void EyeTracker::grayIntegral(Mat src, Mat &dst){ // do gray integral to threshold image
    Mat paintX = Mat::zeros( src.rows, src.cols, CV_8UC1 );
    Mat paintY = Mat::zeros( src.rows, src.cols, CV_8UC1 );
    int* v = new int[src.cols];
    int* h = new int[src.rows];
    uchar* myptr;
    int x,y;
    for( x=0; x<src.cols; x++)
    {
        v[x] = 0;
        for(y=0; y<src.rows; y++)
        {
            myptr = src.ptr<uchar>(y);        //逐行扫描，返回每行的指针
            if( myptr[x] == 0 )
                v[x]++;
        }
    }
    for( x=0; x<src.cols; x++)
    {
        for(y=0; y<v[x]; y++)
        {
            paintX.ptr<uchar>(y)[x] = 255;
        }
    }
    for( x=0; x<src.rows; x++)
    {
        h[x] = 0;
        myptr = src.ptr<uchar>(x);
        for(y=0; y<src.cols; y++)
        {
            if( myptr[y] == 0 )
                h[x]++;
        }
    }
    for( x=0; x<src.rows; x++)
    {
        myptr = paintY.ptr<uchar>(x);
        for(y=0; y<h[x]; y++)
        {
            myptr[y] = 255;
        }
    }
    namedWindow("wnd_X", CV_WINDOW_AUTOSIZE);
    namedWindow("wnd_Y", CV_WINDOW_AUTOSIZE);
    //显示图像
    imshow("wnd_X", paintX);
    imshow("wnd_Y", paintY);
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
    
    if(size - 2*size*filterPercentage - 5 > 0){ // if there is too little points, do not do filtering
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
