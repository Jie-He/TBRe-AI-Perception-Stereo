// ============================================================================
// Definition of the Detector
// Loads the neural network configurations
// ============================================================================
#ifndef _DETECTOR_H_
#define _DETECTOR_H_

#include "cone.h"
#include <opencv2/dnn.hpp>

#include <fstream>
#include <sstream>
#include <iostream>

class Detector{
    public:
        // Need to initialise this network with the method below
        cv::dnn::Net detection_network;
        // Will load the class names into this vector
        std::vector<std::string> classes;

        // Processing arguments 
        const float width  = 416;
        const float height = 416;
        const float detect_thresh = 0.6; 
        const float detect_nms_th = 0.4;
        const float detect_scales = 1/255.0; 

        Detector(int cam_width, int cam_height);
        void frame_process( cv::Mat& frame, std::vector<cone>& clist);
        void drawDetections(cv::Mat& frame, std::vector<cone>& clist);
    private:

        double ratio_w, ratio_h;
        // Default path for neural network configs
        const std::string config_path = "res/cones.cfg";
        const std::string weight_path = "res/cones.weights";
        const std::string cclass_path = "res/cones.names";

        //const std::string config_path = "res/yolov3-tiny.cfg";
        //const std::string weight_path = "res/yolov3-tiny.weights";
        //const std::string cclass_path = "res/coco2.names";

        std::vector<std::string> getOutputNames(const cv::dnn::Net& net);
        void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, 
                            std::vector<cone>& clist);

};

#endif  // _DETECTOR_H_
// ============================================================================