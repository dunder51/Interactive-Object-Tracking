//
//  ObjectTracking.cpp
//  InteractiveObjectTracking
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
        cout << "Right: " << diffX << endl;
    }
    
    if (diffX < 0 && abs(diffX) > wiggleX) {
        cout << "Left: " << diffX << endl;
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
    
    //SiftFeatureDetector detector;
    //Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
    //Ptr<Feature2D> detector = xfeatures2d::SURF::create();
    //Ptr<Feature2D> detector = ORB::create();
    Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
    vector<KeyPoint> keypointsCropped;
    BFMatcher matchmaker;
    vector<DMatch> matches;
    Mat descriptorsCropped;
    
    if ( !cap.isOpened() ) {
        printf("Could not initialize video capture\n");
        return 0;
    }
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
            ROI.copyTo(regionImage);
            resize(regionImage, resizeImageCrop, Size(), .5, .5);
            cvtColor(resizeImageCrop, grayImageCrop, COLOR_BGR2GRAY);
            detector->detect(grayImageCrop, keypointsCropped);
            detector->compute(regionImage, keypointsCropped, descriptorsCropped);
        }
        
        if (croppedCheck && !capture) {
            assert(descriptorsCropped.rows > 0 && descriptorsCropped.cols > 0 && "descriptors empty");
            vector<KeyPoint> keypointsVideo;
            detector->detect(grayImageVid, keypointsVideo);
            Mat descriptorsVideo;
            detector->compute(grayImageVid, keypointsVideo, descriptorsVideo);
            std::vector<std::vector<cv::DMatch>> matches;
            cv::BFMatcher matcher;
            // Find two nearest matches
            matcher.knnMatch(descriptorsVideo, descriptorsCropped, matches, 2);
            assert(displayImage.rows > 0 && displayImage.cols > 0 && "displayImage empty 1");
            assert(resizeImageVid.rows > 0 && resizeImageVid.cols > 0 && "resizeImageVid empty");
            
            vector<cv::DMatch> finalMatches;
            for (int i = 0; i < matches.size(); ++i)
            {
                const float ratio = 0.7; // As in Lowe's paper; can be tuned
                if (matches[i][0].distance < ratio * matches[i][1].distance)
                {
                    finalMatches.push_back(matches[i][0]);
                }
            }
            
            //http://docs.opencv.org/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html?
            assert(displayImage.rows > 0 && displayImage.cols > 0 && "displayImage empty 1");
            assert(resizeImageVid.rows > 0 && resizeImageVid.cols > 0 && "resizeImageVid empty");
            assert(resizeImageCrop.rows > 0 && resizeImageCrop.cols > 0 && "resizeImageCrop empty");
            // Add results to image and save.
            drawMatches(resizeImageVid, keypointsVideo, resizeImageCrop, keypointsCropped, finalMatches, displayImage);
            // Find average location of matches
            int sumX = 0;
            int sumY = 0;
            for(int i = 0; i < finalMatches.size(); i++){
                int matchIndex = finalMatches[i].queryIdx;
                KeyPoint keypointMatch = keypointsVideo[matchIndex];
                Point matchPoint = keypointMatch.pt;
                sumX = sumX + matchPoint.x;
                sumY = sumY + matchPoint.y;
            }
            Point avgPoint;
            if (finalMatches.size() > 0){
                int avgX = abs(floor(sumX/finalMatches.size()));
                int avgY = abs(floor(sumY/finalMatches.size()));
                avgPoint = Point(avgX, avgY);
            }
            targetLocation(avgPoint, displayImage);
        }
        showImage();
    }
}
