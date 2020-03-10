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

#include <opencv2/highgui.hpp>

#include "Detector.h" // <- the other opencv includes are here
#include "cone.h"

// ZED
#include <sl/Camera.hpp>

// ============================================================================
// Create window
static const std::string iWName = "Input";
static const std::string oWName = "Result";
static const std::string dWName = "Depth";
// ============================================================================

using namespace cv;
cv::Mat slMat2cvMat(sl::Mat& input);

void approx_distance(sl::Mat point_cloud, std::vector<cone>& clist){
    sl::float4 point3D;
                    
    // get dept information on each cone
    for (size_t i = 0; i < clist.size(); i++){
        point_cloud.getValue(clist[i].cone_centre.x,
                             clist[i].cone_centre.y, &point3D);

        // calculate the mean between few points 
        
        clist[i].cone_point_dist = \
        sqrt(point3D.x*point3D.x + point3D.y*point3D.y + point3D.z*point3D.z);
            
        std::cout << clist[i].cone_class << " : "
                  << clist[i].cone_point_dist << std::endl;
    }
}

void zed_mode(sl::Camera& zed){
    std::cout << "ZED detected" << std::endl;

    sl::RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = sl::SENSING_MODE::STANDARD;

    // prepare new image size 
    sl::Resolution image_size = 
    zed.getCameraInformation().camera_resolution;
    int new_width = image_size.width / 2;
    int new_height= image_size.height/ 2;

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

    //sl::Mat image_zedr(new_width, new_height, sl::MAT_TYPE::U8_C4);
    Mat image_ocvr;// = slMat2cvMat(image_zedr);

    sl::Mat depth_image_zed(new_width, new_height, sl::MAT_TYPE::U8_C4);
    Mat depth_image_ocv = slMat2cvMat(depth_image_zed);
    sl::Mat point_cloud;
    Mat point_cloud_ocv = slMat2cvMat(point_cloud);

    while (true){
        if(zed.grab(runtime_parameters) == sl::ERROR_CODE::SUCCESS){
        // get left frame, depth image in half res
        zed.retrieveImage(image_zedl, sl::VIEW::LEFT,
                                      sl::MEM::CPU, new_image_size);

        //zed.retrieveImage(image_zedr, sl::VIEW::RIGHT,
        //                              sl::MEM::CPU, new_image_size);

        zed.retrieveImage(depth_image_zed, sl::VIEW::DEPTH, 
                                           sl::MEM::CPU, new_image_size);


        // Retrieve the RGBA point cloud in half res
        zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA,
                                         sl::MEM::CPU, new_image_size);

        // display image
        cv::imshow(iWName, image_ocvl);
        //cv::imshow(oWName, image_ocvr);
        cvtColor(image_ocvl, image_ocvr, COLOR_BGRA2BGR);

        // process this image
        std::vector<cone> clist;
        detector.frame_process(image_ocvr, clist);
        
        // compute distance using point cloud
        approx_distance(point_cloud, clist);

         // draw the bounding boxes
        detector.drawDetections(image_ocvr, clist);

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
        init_params.coordinate_units = sl::UNIT::METER;
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