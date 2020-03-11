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


#include <opencv2/highgui.hpp>

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

const float dist_right = 120.0; // In milimeters
const float focal_lent = 2.8;
const float pixle_size = 0.002;
// ============================================================================

using namespace cv;
cv::Mat slMat2cvMat(sl::Mat& input);

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
    std::cout << "valid counts: " << valid_count << std::endl;
    pt.x = x; pt.y = y; pt.z = z;
}

void approx_distance(sl::Mat& point_cloud, std::vector<cone>& clist,
                     cv::Mat& leftframe, cv::Mat& rightFrame){
    sl::float4 point3D;
    static int sample_size = 100;
    
    // crop out the cone in the left image
    Mat crop;
    std::vector<Point> whitePix;
    
    // get dept information on each cone
    for (size_t i = 0; i < clist.size(); i++){
        //crop = Mat(leftframe, clist[i].cone_box);
        crop = leftframe( Rect(0, 0, leftframe.cols, leftframe.rows)
                          & clist[i].cone_box);
                          
        // convert to HSV channel
        cvtColor(crop, crop, COLOR_BGRA2RGB);
        cvtColor(crop, crop, COLOR_RGB2HSV);
        // produce black and white pic
        inRange( crop, Scalar(14, 125, 0), Scalar(255, 255, 255), crop);
        // count the number of white pixles
        cv::findNonZero(crop, whitePix);

        point_cloud.getValue(clist[i].cone_centre.x,
                             clist[i].cone_centre.y, &point3D);

        // apply the offset


        // sample 100 points frin wpix;
        // if no more than 100 points use the list
        if (whitePix.size() > sample_size){
            std::random_shuffle(whitePix.begin(), whitePix.end());
            avg_list_point(point_cloud, point3D, whitePix, sample_size,
                           clist[i].cone_box.x, clist[i].cone_box.y);
        }else{
            avg_list_point(point_cloud, point3D, whitePix, whitePix.size(),
                           clist[i].cone_box.x, clist[i].cone_box.y);
        }

        // average the first 100 points (some might be NaN)
        imshow(sWName, crop);
        // calc centre point in right frame

        // calculate the mean between few points 
        clist[i].cone_point_dist = \
        sqrt(point3D.x*point3D.x + point3D.y*point3D.y + point3D.z*point3D.z);
            
        std::cout << clist[i].cone_class << " : "
                  << clist[i].cone_point_dist << std::endl;

        std::cout << "xyz: " << point3D.x << "  #  "
                  << point3D.y << "  #  " << point3D.z << std::endl; 

        // approx this x in the other frame
        int rx; // same y
        float m = (point3D.z) / (point3D.x - dist_right);
        rx = ((focal_lent / m) / pixle_size ) + 960; // need to add half the screen width
        //draw a circle in the right frame of radius 10
        circle(rightFrame, Point(rx, clist[i].cone_centre.y), 10, Scalar(0, 255, 0), 2);
    }
}

void zed_mode(sl::Camera& zed){
    std::cout << "ZED detected" << std::endl;

    sl::RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = sl::SENSING_MODE::STANDARD;

    // prepare new image size 
    sl::Resolution image_size = 
    zed.getCameraInformation().camera_resolution;

    //take this half res off for now
    int new_width = image_size.width ;
    int new_height= image_size.height;

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