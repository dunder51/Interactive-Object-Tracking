//
//  ObjectTracking.cpp
//  InteractiveObjectTracking-template matching
//
//  Created by Duncan Calder on 11/7/18.
//  Copyright © 2018 Duncan Calder. All rights reserved.
//

#include "ObjectTracking.hpp"

using namespace cv;
using namespace std;

const char* windowName = "Image Window";

Mat img;
Mat src;
Mat ROI;
Mat displayImage;
Rect selectRect;
Point P1(0,0);
Point P2(0,0);
bool clicked = false;
bool capture = false;
bool croppedCheck = false;
bool newCrop = false;

Rect checkBoundary(Mat img, Rect selectRect) {
    //check if rectangle exceeds image boundary
    if(selectRect.width > img.cols-selectRect.x)
        selectRect.width = img.cols-selectRect.x;
    
    if(selectRect.height > img.rows-selectRect.y)
        selectRect.height = img.rows-selectRect.y;
    
    if(selectRect.x < 0)
        selectRect.x = 0;
    
    if(selectRect.y < 0)
        selectRect.height = 0;
    return selectRect;
}

void showImage() {
    newCrop = false;
    src.copyTo(img);
    if (capture){
        checkBoundary(img, selectRect);
        if(selectRect.width > 0 && selectRect.height > 0){
            ROI = src(selectRect);
            imshow("cropped",ROI);
        }
        
        rectangle(img, selectRect, Scalar(0, 255 ,0), 1, 8, 0 );
        croppedCheck = true;
        newCrop = true;
    }
    if (croppedCheck && !capture){
        //cout << "show displayImage" << endl;
        imshow(windowName, displayImage);
    }
    else {
        //cout << "show regular image" << endl;
        imshow(windowName,img);
    }
}

void keypointsCrop(Ptr<Feature2D> detector, Mat regionImage, vector<KeyPoint> keypoints, Mat descriptors){
    cout << "here" << endl;
    Mat resizeImage;
    Mat grayImage;
    resize(regionImage, resizeImage, Size(), .5, .5);
    cvtColor(resizeImage, grayImage, COLOR_BGR2GRAY);
    detector->detect(grayImage, keypoints);
    detector->compute(resizeImage, keypoints, descriptors);
}

void targetLocation(Point location, Mat image){
    int width = image.cols;
    int height = image.rows;
    
    Point center = Point(floor(width/2), floor(height/2));
    
    //cout << width << " " << height << " " << center << endl;
    
    int diffX = center.x - location.x;
    int diffY = center.y - location.y;
    int percentWiggle = 10;
    int wiggleX = floor(width / percentWiggle);
    int wiggleY = floor(height / percentWiggle);
    
    if (diffX > 0 && diffX > wiggleX) {
        cout << "Left: " << diffX << endl;
    }
    
    if (diffX < 0 && abs(diffX) > wiggleX) {
        cout << "Right: " << diffX << endl;
    }
    
    
    if (diffY < wiggleY && diffX < wiggleX && diffY < 0 && diffX < 0) {
        //cout << "No Move" << endl;
        //No move
    }
}

void onMouse(int event, int x, int y, int f, void*){
    
    switch(event){
        case  CV_EVENT_LBUTTONDOWN:
            clicked=true;
            P1.x = x;
            P1.y = y;
            P2.x = x;
            P2.y = y;
            break;
            
        case  CV_EVENT_LBUTTONUP:
            P2.x = x;
            P2.y = y;
            clicked = false;
            break;
            
        case  CV_EVENT_MOUSEMOVE:
            if(clicked){
                P2.x = x;
                P2.y = y;
            }
            break;
            
        default:
            break;
    }
    
    if(clicked){
        if(P1.x > P2.x) {
            selectRect.x = P2.x;
            selectRect.width = P1.x-P2.x; }
        else {
            selectRect.x = P1.x;
            selectRect.width = P2.x-P1.x;
        }
        
        if(P1.y > P2.y){
            selectRect.y = P2.y;
            selectRect.height = P1.y-P2.y;
        }
        else {
            selectRect.y = P1.y;
            selectRect.height = P2.y-P1.y;
        }
        
    }
    if (capture) {
        showImage();
    }
}

int main(int argc, char* argv[]) {
    namedWindow("Image Window", 1);
    
    VideoCapture cap;
    cap.open(0);
    Mat firstRead;
    cap.read(firstRead);
    resize(firstRead, src, Size(), .5, .5);
    setMouseCallback(windowName, onMouse);
    
    if ( !cap.isOpened() ) {
        printf("Could not initialize video capture\n");
        return 0;
    }
    //I'm realizing that i need to keep these open because i want the cropped image to not change unless it is changed so i can't have it resetting between loops
    Mat resizeImageCrop;
    Mat grayImageCrop;
    Mat regionImage;
    
    while (1 == 1) {
        
        Mat originalImage;
        cap.read(originalImage);
        Mat resizeImageVid;
        Mat grayImageVid;
        resize(originalImage, resizeImageVid, Size(), .5, .5);
        cvtColor(resizeImageVid, grayImageVid, COLOR_BGR2GRAY);
        resizeImageVid.copyTo(src);
        resizeImageVid.copyTo(displayImage);
        
        Point matchLoc;
        
        char key = waitKey(33);
        if (key == 'q') {
            break;
        }
        if (key == 'p') {
            capture = true;
            croppedCheck = false;
        }
        if (key == 's') {
            capture = false;
        }
        
        
        if (newCrop && !capture) {
            //cout << "keypointsCrop" << endl;
            ROI.copyTo(regionImage);
            resize(regionImage, resizeImageCrop, Size(), .5, .5);
            cvtColor(resizeImageCrop, grayImageCrop, COLOR_BGR2GRAY);
            
            
        }
        
        if (croppedCheck && !capture) {
            //cout << "descriptors match" << endl;
            
            Mat result;
            matchTemplate( resizeImageVid, resizeImageCrop, result, TM_CCORR_NORMED);
            normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
            /// Localizing the best match with minMaxLoc
            double minVal; double maxVal; Point minLoc; Point maxLoc;
            Point matchLoc;
            
            minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
            // Add results to image and save.
            assert(displayImage.rows > 0 && displayImage.cols > 0 && "displayImage empty 1");
            assert(resizeImageVid.rows > 0 && resizeImageVid.cols > 0 && "resizeImageVid empty");
            
            //assert(displayImage.rows > 0 && displayImage.cols > 0 && "displayImage empty 1");
            //cout << "resizeVid: " << resizeImageVid.rows << endl;
            assert(resizeImageVid.rows > 0 && resizeImageVid.cols > 0 && "resizeImageVid empty");
            assert(resizeImageCrop.rows > 0 && resizeImageCrop.cols > 0 && "resizeImageCrop empty");
            
            matchLoc = maxLoc;
            rectangle( displayImage, matchLoc, Point( matchLoc.x + resizeImageCrop.cols , matchLoc.y + resizeImageCrop.rows ), Scalar::all(0), 2, 8, 0 );
            //rectangle( result, matchLoc, Point( matchLoc.x + resizeImageCrop.cols , matchLoc.y + resizeImageCrop.rows ), Scalar::all(0), 2, 8, 0 );
            //cout << matchLoc << endl;
            targetLocation(matchLoc, displayImage);
            
        }
        showImage();
        //targetLocation(matchLoc, resizeImageCrop);
        //imshow(windowName,src);
    }
}
