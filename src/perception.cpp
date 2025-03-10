// ============================================================================
// TBRE PERCEPTION PIPELINE
// Keep each line under 80 chars
// ============================================================================
// Uses yolov3.cfg and weights to detect cones from stereo cam
//  -   by default, for ease of usage
//  -   place the .cfg and .weight into "res/" where the executable is
//  -   to improve the weights, have a look at the darknet documentation
// ============================================================================

// C++
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <algorithm>    // std::random_shuffle
#include <cstdlib>      // std::rand

// Opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
// SIFT descriptor // extractor
#include <opencv2/xfeatures2d.hpp>

#include "Detector.h" // <- the other opencv includes are here
#include "cone.h"

// ZED
#include <sl/Camera.hpp>

// ============================================================================
// Create window
static const std::string iWName = "Left";
static const std::string oWName = "Right";
static const std::string dWName = "Depth";
static const std::string sWName = "crops";

// ZED camera properties
const float dist_right = 120.0; // In milimeters
const float focal_lent = 2.8;
const float pixle_size = 0.004;

// SIFT setting
const int minHessian = 400;
const float ratio_thresh  = 0.7f;
// ============================================================================

using namespace cv;
cv::Mat slMat2cvMat(sl::Mat& input);

void sift_triangulate(Mat imgl, Mat imgr, cone c, Point lxy, Point rxy){
    Ptr<xfeatures2d::SIFT> sift_detector = \
    xfeatures2d::SIFT::create(minHessian);

    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptor1, descriptor2;

    sift_detector->detectAndCompute( imgl, noArray(), keypoints1, descriptor1);
    sift_detector->detectAndCompute( imgr, noArray(), keypoints2, descriptor2);

    // Match the descriptor with FLANN based matcher
    Ptr<DescriptorMatcher> matcher = \
    DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

    std::vector<std::vector<DMatch>> knn_matches;
    
    matcher->knnMatch( descriptor1, descriptor2, knn_matches, 2);

    // Filter with Lowe's ratio test
    std::vector<DMatch> good_matches;

    Point2f point1, point2;

    for(size_t i = 0; i < knn_matches.size(); i++){
        if(knn_matches[i][0].distance < \
        ratio_thresh * knn_matches[i][1].distance){
            good_matches.push_back(knn_matches[i][0]);    
        }
    }

    // Draw matches
    Mat img_matches;
    drawMatches(imgl, keypoints1, imgr, keypoints2, good_matches,
                img_matches, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    imshow("MATCHES", img_matches);
}

void avg_list_point(sl::Mat& point_cloud, sl::float4& pt,
                    std::vector<Point>& sample, int count, int ofx, int ofy){
    float x, y, z;
    x = 0; y = 0; z = 0;
    int valid_count = 0;

    for(int i = 0; i < count; i++){
        point_cloud.getValue(sample[i].x + ofx, sample[i].y + ofy, &pt);
        if (!std::isnan(pt.z) && !std::isnan(pt.x) && !std::isnan(pt.y)){
            x += pt.x;
            y += pt.y;
            z += pt.z;
            valid_count ++;
        }
    }
    x = x / valid_count;
    y = y / valid_count;
    z = z / valid_count;
    //std::cout << "valid counts: " << valid_count << std::endl;
    pt.x = x; pt.y = y; pt.z = z;
}

void approx_distance(sl::Mat& point_cloud, std::vector<cone>& clist,
                     cv::Mat& leftframe, cv::Mat& rightFrame){
    sl::float4 point3D;
    static int sample_size = 50;
    
    // crop out the cone in the left image
    Mat leftcrop, rightcrop, crophsv;
    std::vector<Point> whitePix;
    
    // get dept information on each cone
    for (size_t i = 0; i < clist.size(); i++){
        //crop = Mat(leftframe, clist[i].cone_box);
        leftcrop = leftframe( Rect(0, 0, leftframe.cols, leftframe.rows)
                          & clist[i].cone_box);
                          
        // convert to HSV channel
        cvtColor(leftcrop, crophsv, COLOR_BGRA2RGB);
        cvtColor(crophsv , crophsv, COLOR_RGB2HSV);
        // produce black and white pic
        inRange( crophsv, Scalar(14, 125, 0), Scalar(255, 255, 255), crophsv);
        // count the number of white pixles
        cv::findNonZero(crophsv, whitePix);

        point_cloud.getValue(clist[i].cone_centre.x,
                             clist[i].cone_centre.y, &point3D);

        // apply the offset
        // calculate the mean between few points 
        clist[i].cone_point_dist = \
        sqrt(point3D.x*point3D.x + point3D.y*point3D.y + point3D.z*point3D.z);

        // sample 100 points frin wpix;
        //if no more than 100 points use the list
        if (whitePix.size() > sample_size){
            std::random_shuffle(whitePix.begin(), whitePix.end());
            avg_list_point(point_cloud, point3D, whitePix, sample_size,
                           clist[i].cone_box.x, clist[i].cone_box.y);
        }else{
             avg_list_point(point_cloud, point3D, whitePix, whitePix.size(),
                            clist[i].cone_box.x, clist[i].cone_box.y);
        }

        // average the first 50 points (some might be NaN)
        imshow(sWName, crophsv);

        std::cout << "Class: " << clist[i].cone_class << ", "
                  << "Predn: " << clist[i].cone_accuy << ", "
                  << "Distn: " << clist[i].cone_point_dist << ", "
                  << "Trign: " << clist[i].cone_trig_dist << std::endl;  

        if( !std::isnan(clist[i].cone_point_dist)){
            // approx this x in the other frame
            int rx; // same y
            float m = (point3D.z) / (point3D.x - dist_right);
            rx = ((focal_lent / m) / pixle_size ) + (rightFrame.cols / 2); 
            // need to add half the screen width

            // define a rectangle that is 2x as wide
            Rect right_box = Rect(rx - (clist[i].cone_box.width), 
                    clist[i].cone_box.y, clist[i].cone_box.width * 2, 
                    clist[i].cone_box.height);

            rightcrop = rightFrame(Rect( 0, 0, rightFrame.cols, rightFrame.rows) 
                                    & right_box );

            sift_triangulate(leftcrop, rightcrop, clist[i], 
                            clist[i].cone_box.tl(),
                            right_box.tl());

            // draw a rectangle that is 2x as wide
            rectangle(rightFrame, right_box, Scalar(0, 255, 0), 1);
            //draw a circle in the right frame of radius 10
            circle(rightFrame, Point(rx, clist[i].cone_centre.y), 10, 
                                     Scalar(0, 255, 0), 1);  
        }
    }
    std::cout << "================== DIVIDER ====================" << std::endl;    
}

void zed_mode(sl::Camera& zed){
    std::cout << "ZED detected" << std::endl;

    sl::RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = sl::SENSING_MODE::STANDARD;

    // prepare new image size 
    sl::Resolution image_size = 
    zed.getCameraInformation().camera_resolution;

    //take this half res off for now
    int new_width = image_size.width /2;
    int new_height= image_size.height/2;

    // initialise detector
    Detector detector = Detector(new_width, new_height);

    std::cout << "Camera Resolution: " << new_width
              << " x " << new_height << std::endl;

    sl::Resolution new_image_size(new_width, new_height);

    // To share data between sl::Mat and cv::Mat, use slMat2cvMat()
    // Only the headers and pointer to the sl::Mat are 
    //copied, not the data itself
    sl::Mat image_zedl(new_width, new_height, sl::MAT_TYPE::U8_C4);
    Mat image_ocvl = slMat2cvMat(image_zedl);

    sl::Mat image_zedr(new_width, new_height, sl::MAT_TYPE::U8_C4);
    Mat image_ocvr = slMat2cvMat(image_zedr);

    sl::Mat depth_image_zed(new_width, new_height, sl::MAT_TYPE::U8_C4);
    Mat depth_image_ocv = slMat2cvMat(depth_image_zed);

    sl::Mat point_cloud;
    Mat point_cloud_ocv = slMat2cvMat(point_cloud);

    while (true){
        if(zed.grab(runtime_parameters) == sl::ERROR_CODE::SUCCESS){
        // get left frame, depth image in half res
        zed.retrieveImage(image_zedl, sl::VIEW::LEFT,
                                      sl::MEM::CPU, new_image_size);

        zed.retrieveImage(image_zedr, sl::VIEW::RIGHT,
                                      sl::MEM::CPU, new_image_size);

        zed.retrieveImage(depth_image_zed, sl::VIEW::DEPTH, 
                                           sl::MEM::CPU, new_image_size);

        // Retrieve the RGBA (MAYBE CHANGE TO HS)point cloud in half res
        zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA,
                                         sl::MEM::CPU, new_image_size);

        // process this image
        std::vector<cone> clist;
        detector.frame_process(image_ocvl, clist);
        
        // compute distance using point cloud
        approx_distance(point_cloud, clist, image_ocvl, image_ocvr);

         // draw the bounding boxes
        detector.drawDetections(image_ocvl, clist);

        // display image
        cv::imshow(iWName, image_ocvl);
        cv::imshow(oWName, image_ocvr);
        cv::imshow(dWName, depth_image_ocv);

        if( cv::waitKey(10) >= 0 ) break;
        }
    }
    zed.close();
}

void cam_mode(VideoCapture& cap){
    // initialise detector
    Detector detector = Detector(cap.get(CAP_PROP_FRAME_WIDTH),
                                 cap.get(CAP_PROP_FRAME_HEIGHT));

    Mat frame, result;
    while(waitKey(10)){
        cap >> frame;    
        
        result = frame.clone();
        if (frame.empty()){
            waitKey();
            break;
        }


        std::vector<cone> clist;
        detector.frame_process(frame, clist);
        detector.drawDetections(result, clist);
            
        imshow(iWName, frame);
        imshow(oWName, result);
    }
}

int main (int argc, char** argv){

    namedWindow(dWName, WINDOW_AUTOSIZE);
    namedWindow(iWName, WINDOW_AUTOSIZE);
    namedWindow(oWName, WINDOW_AUTOSIZE);
    namedWindow(sWName, WINDOW_AUTOSIZE);

    // Activate the video cam
    VideoCapture cap;
    
    // If you supply path to a video or image 
    bool has_zed = false;
    if (argc == 2){
        cap.open(argv[1]);
        cam_mode(cap);
    }else{ // Open the camera
        sl::Camera zed;

        // set configs
        sl::InitParameters init_params;
        init_params.camera_resolution = sl::RESOLUTION::HD1080;
        init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
        init_params.coordinate_units = sl::UNIT::MILLIMETER;
        init_params.camera_fps = 30;

        // try to open
        sl::ERROR_CODE err = zed.open(init_params);
        if ( err == sl::ERROR_CODE::SUCCESS ){
            // ZED hooked up.
            zed_mode(zed);
        }else{
            std::cout << "Unable to initialise ZED: " 
            << sl::toString(err).c_str()
            << std::endl;

            // No zed. launch built camera instead
            cap.open(0);
            cam_mode(cap);
        }
    }
    
    
    return 0;
}

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(sl::Mat& input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }

    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 
    // pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return Mat(input.getHeight(), input.getWidth(), cv_type, 
                    input.getPtr<sl::uchar1>(sl::MEM::CPU));
}
// ============================================================================