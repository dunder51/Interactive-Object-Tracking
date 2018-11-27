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

int main(int argc, char* argv[])
{
    namedWindow("Image Window", 1);
    Mat image1;
    image1 = imread(argv[1]);
    Mat grayImage1;
    cvtColor(image1, grayImage1, COLOR_BGR2GRAY);
    
    VideoCapture cap;
    cap.open(0);
    
    Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
    vector<KeyPoint> keypoints1;
    detector->detect(grayImage1, keypoints1);
    Mat descriptors1;
    detector->compute(image1, keypoints1, descriptors1);
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
        Mat resizedImage;
        cap.read(originalImage);
        originalImage.copyTo(resizedImage);
        resize(originalImage, resizedImage, Size(), .5, .5);
        Mat grayImage;
        cvtColor(resizedImage, grayImage, COLOR_BGR2GRAY);
        
         //SiftFeatureDetector detector;
         //Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create();
         //Ptr<Feature2D> detector = xfeatures2d::SURF::create();
         //Ptr<Feature2D> detector = ORB::create();
        detector->detect(grayImage, keypoints2);
        Mat descriptors2;
        detector->compute(grayImage, keypoints2, descriptors2);
        matchmaker.match(descriptors2, descriptors1, matches);
         
         // Add results to image and save.
         /*drawKeypoints(resizedImage, keypoints2, displayImage, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        */
        
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
        drawMatches(resizedImage, keypoints2, image1, keypoints1, finalMatches, displayImage);
        
        drawMatches(resizedImage, keypoints2, image1, keypoints1, finalMatches, displayImage);
        imshow("Image Window", displayImage);
        char key = waitKey(33);
        if (key == 'q')
        {
            break;
        }
        if (key == 'p')
        {
            
        }
    }
}
