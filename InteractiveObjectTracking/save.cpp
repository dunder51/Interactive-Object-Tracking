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
Rect selectRect;
Point P1(0,0);
Point P2(0,0);
bool clicked = false;
bool capture = false;

Rect checkBoundary(Mat img, Rect selectRect){
    //check rectangle exceed image boundary
    if(selectRect.width>img.cols-selectRect.x)
        selectRect.width=img.cols-selectRect.x;
    
    if(selectRect.height>img.rows-selectRect.y)
        selectRect.height=img.rows-selectRect.y;
    
    if(selectRect.x<0)
        selectRect.x=0;
    
    if(selectRect.y<0)
        selectRect.height=0;
    return selectRect;
}

Mat readyImage(Mat img){
    Mat grayImage;
    resize(img, img, Size(), .5, .5);
    cvtColor(img, grayImage, COLOR_BGR2GRAY);
    return grayImage;
}

void showImage(){
    img=src.clone();
    if (capture){
        checkBoundary(img, selectRect);
        if(selectRect.width>0&&selectRect.height>0){
            ROI = src(selectRect);
            imshow("cropped",ROI);
        }
    
    rectangle(img, selectRect, Scalar(0,255,0), 1, 8, 0 );
    }
    imshow(windowName,img);
}

void onMouse( int event, int x, int y, int f, void* ){
    
    switch(event){
        case  CV_EVENT_LBUTTONDOWN:
            clicked=true;
            P1.x=x;
            P1.y=y;
            P2.x=x;
            P2.y=y;
            break;
            
        case  CV_EVENT_LBUTTONUP:
            P2.x=x;
            P2.y=y;
            clicked=false;
            break;
            
        case  CV_EVENT_MOUSEMOVE:
            if(clicked){
                P2.x=x;
                P2.y=y;
            }
            break;
            
        default:
            break;
    }
    
    if(clicked){
        if(P1.x>P2.x) {
            selectRect.x=P2.x;
            selectRect.width=P1.x-P2.x; }
        else {
            selectRect.x=P1.x;
            selectRect.width=P2.x-P1.x;
        }
        
        if(P1.y>P2.y){
            selectRect.y=P2.y;
            selectRect.height=P1.y-P2.y;
        }
        else {
            selectRect.y=P1.y;
            selectRect.height=P2.y-P1.y;
        }
        
    }
    if (capture) {
        showImage();
    }
}

int main(int argc, char* argv[])
{
    namedWindow("Image Window", 1);
    
    /*Mat image1;
    image1 = imread(argv[1]);
    Mat grayImage1;
    cvtColor(image1, grayImage1, COLOR_BGR2GRAY);
    */
    VideoCapture cap;
    cap.open(0);
    Mat firstRead;
    cap.read(firstRead);
    resize(firstRead, src, Size(), .5, .5);
    setMouseCallback(windowName, onMouse);
    
    Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
    vector<KeyPoint> keypoints1;
    /*
    detector->detect(grayImage1, keypoints1);
    Mat descriptors1;
    detector->compute(image1, keypoints1, descriptors1);
    */
    BFMatcher matchmaker;
    vector<DMatch> matches;
    
    
    if (!cap.isOpened())
    {
        printf("Could not initialize video capture\n");
        return 0;
    }
    while (1 == 1)
    {
        
        vector<KeyPoint> keypoints2;
        Mat originalImage;
        Mat displayImage;
        cap.read(originalImage);
        Mat processedImage = readyImage(originalImage);
        processedImage.copyTo(src);
        showImage();
        
        if ((ROI.rows <= 0 || ROI.cols <= 0) && !capture) {
            
            //SiftFeatureDetector detector;
            //Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
            //Ptr<Feature2D> detector = xfeatures2d::SURF::create();
            //Ptr<Feature2D> detector = ORB::create();
            detector->detect(processedImage, keypoints2);
            Mat descriptors2;
            detector->compute(processedImage, keypoints2, descriptors2);
            matchmaker.match(descriptors2, descriptors1, matches);
         
            // Add results to image and save.
            drawKeypoints(originalImage, keypoints2, displayImage, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        
            int threshold = 120;
        
            vector<DMatch> finalMatches;
            for (int i = 0; i<matches.size(); i++)
            {
                if (matches[i].distance < threshold)
                {
                    finalMatches.push_back(matches[i]);
                
                }
            
            }
            //http://docs.opencv.org/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html?
            drawMatches(originalImage, keypoints2, image1, keypoints1, finalMatches, displayImage);
        
            drawMatches(originalImage, keypoints2, image1, keypoints1, finalMatches, displayImage);
        }
        //imshow("Image Window", displayImage);
        //imshow(windowName,src);
        char key = waitKey(33);
        if (key == 'q')
        {
            break;
        }
        if (key == 'p')
        {
            capture = true;
        }
        if (key == 's')
        {
            capture = false;
        }
    }
}
