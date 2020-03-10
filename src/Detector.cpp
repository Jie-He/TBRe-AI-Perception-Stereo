// ============================================================================
// Implementation of the Detector
// ============================================================================
#include "Detector.h"

using namespace cv;
using namespace dnn;

// ============================================================================
/**
 * Main constructor of the detector class
 * Loads the network configuration
 * @param cam_width the width of the input image
 * @param cam_height -  height -  -    -     -
 */
Detector::Detector(int cam_width, int cam_height){
    // Load the class names from cclass_path
    std::ifstream ifs(cclass_path.c_str());
    if(!ifs.is_open())
        CV_Error(Error::StsError, "File " + cclass_path + " not found.");
    
    std::string line;
    while (std::getline(ifs, line)){
        classes.push_back(line);
    }

    // Load the neural network model
    detection_network = readNet(config_path, weight_path);
    detection_network.setPreferableBackend(DNN_BACKEND_OPENCV);
    detection_network.setPreferableTarget(DNN_TARGET_CPU);  

    // calculate the ratio from input to network
    // for bounding box purposes
    ratio_h = cam_height / height;
    ratio_w = cam_width  / width;

    std::cout << "Network Initialised!" << std::endl;
}
// ============================================================================
/**
 * Perform object detection on a given frame
 * @param frame the input image to be processed
 * @param clist where the list of the detected cones will be stored
 */
void Detector::frame_process(Mat& frame, std::vector<cone>& clist){
    // Scale the input frame into network size
    static Mat res, blob;
    
    resize(frame, res, Size(width, height));
    blobFromImage(res, blob, detect_scales, Size(width, height),
                    Scalar(0,0,0), true, false);

    // Pass the blob to network
    detection_network.setInput(blob);

    std::vector<Mat> outs;
    detection_network.forward(outs, getOutputNames(detection_network));

    // Filter out the bounding box with low confidence
    postprocess(res, outs, clist);

    // apply ratio change to each cone
    for (size_t i = 0; i < clist.size(); i++){
            clist[i].cone_centre.x *= ratio_w;
            clist[i].cone_centre.y *= ratio_h;
            clist[i].cone_box.x *= ratio_w;
            clist[i].cone_box.y *= ratio_h;
            clist[i].cone_box.width *= ratio_w;
            clist[i].cone_box.height*= ratio_h;        
    }

    // Can do some efficiency analysis here
    std::vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = detection_network.getPerfProfile(layersTimes) / freq;
    std::string label = format("Inference time: %.2f ms", t);
    std::string count = format("Number of cones: %ld", clist.size());
    putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, 
            Scalar(0, 255, 0));
    putText(frame, count, Point(0, 30), FONT_HERSHEY_SIMPLEX, 0.5,
            Scalar(0, 255, 0));
}

/**
 * In short, matches the output layer with the correct class name
 * @param net the network just did your work
 * @return names of the classes
 */
std::vector<String> Detector::getOutputNames(const Net& net){
    static std::vector<String> names;
    if (names.empty()){
        std::vector<int> outLayers = net.getUnconnectedOutLayers();

        std::vector<String> layersNames = net.getLayerNames();

        names.resize(outLayers.size());
        for(size_t i = 0; i < outLayers.size(); i++){
            names[i] = layersNames[outLayers[i] - 1];
        }
    }
    return names;
}

/**
 * Should not be able to access this function outside of detector
 * Performs filtering to remove unwanted bounding boxes 
 * (i.e when over lapping boxes occur)
 * Produces a list of cones from raw output of the network
 * @param frame the input image to be processed
 * @param outs  the output of the network
 */ 
void Detector::postprocess(Mat& frame, const std::vector<Mat>& outs, 
    std::vector<cone>& clist){
    std::vector<int>    classID;
    std::vector<float>  confidence;
    std::vector<Rect>   bounding;
    std::vector<Point>  centreP;

    for (size_t i = 0; i < outs.size(); i++){
        // For each cone detection, filter out the ones with less confidence 
        // Than previously defined
        float* data = (float*) outs[i].data;
        for(int j = 0; j < outs[i].rows; j++, data+= outs[i].cols){
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIDPoint;
            double conf;

            minMaxLoc(scores, 0, &conf, 0, &classIDPoint);

            if (conf > detect_thresh){
                // If its above the threshold, calculate the bounding box
                int centreX = (int)(data[0] * frame.cols);
                int centreY = (int)(data[1] * frame.rows);
                int b_width = (int)(data[2] * frame.cols);
                int b_heigh = (int)(data[3] * frame.rows);

                int b_x = centreX - b_width/2;
                int b_y = centreY - b_heigh/2;

                classID.push_back(classIDPoint.x);
                confidence.push_back((float)conf);
                bounding.push_back(Rect(b_x, b_y, b_width, b_heigh));
                centreP.push_back(Point(centreX, centreY));
            }
        }
    }

    // Perform non maximum suppression to remove redundant overlapping boxes
    std::vector<int> indices;
    NMSBoxes(bounding, confidence, detect_thresh, detect_nms_th, indices);

    for(size_t i = 0; i < indices.size(); i++){
        int idx = indices[i];

        cone c;
        c.cone_class = classID[idx];
        c.cone_accuy = (int)(confidence[idx] * 100);
        c.cone_box   = bounding[idx];
        c.cone_centre= centreP[idx];
        c.cone_point_dist = 999; // default far away?
        clist.push_back(c);
    }
}

/** this draws a list of detections
 * @param frame is where its going to draw on
 * @param clist is the list of cones its going to draw on
 */
void Detector::drawDetections(Mat& frame, std::vector<cone>& clist){
    for (size_t i = 0; i < clist.size(); i++){
        // Define the box colour;
        Scalar colour; 
        // Draw identical colour box for each
        if (clist[i].cone_class == 0) // Yellow || YY
            colour = Scalar(   0, 255, 255);
        else if (clist[i].cone_class == 1)
            colour = Scalar( 255,  64,   0);
        else if (clist[i].cone_class == 2 || clist[i].cone_class == 3)
            colour = Scalar(   0, 140, 255);
        else
            colour = Scalar( 0, 0, 255);
        rectangle(frame, clist[i].cone_box, colour);
        circle(frame, clist[i].cone_centre, clist[i].cone_box.width / 2
                , colour, 1);
    

        // Grab the label
        std::string label = format("%d - %.2f (unit)", 
                                clist[i].cone_accuy,
                                clist[i].cone_point_dist);
        if(!classes.empty()){
            CV_Assert(clist[i].cone_class < (int)classes.size());
            label = classes[clist[i].cone_class] + ":" + label;
        }

        // Display the labels
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SCRIPT_SIMPLEX, 
                                    0.5, 1, &baseLine);
        clist[i].cone_box.y = max(clist[i].cone_box.y, labelSize.height);
        putText(frame, label, clist[i].cone_box.tl(), 
                FONT_HERSHEY_SIMPLEX, 0.5, colour);
    }
}
// ============================================================================