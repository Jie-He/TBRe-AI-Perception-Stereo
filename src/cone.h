// ============================================================================
// Definitioin of the Cone class
// Contains the following informations
//      -   cone class. (int) e.g 0 (Yellow)
//      -   accuracy of detection (int)
//      -   bounding box (Rect)
// ============================================================================
#ifndef _CONE_H_
#define _CONE_H_

#include <opencv2/imgproc.hpp>

class cone{
    public:
        int cone_class;
        int cone_accuy;
        cv::Point cone_centre;
        cv::Rect cone_box;
        float cone_point_dist;
        float cone_trig_dist;
};

#endif  // _CONE_H_
// ============================================================================