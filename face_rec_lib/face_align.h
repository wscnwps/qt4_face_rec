#ifndef FACE_ALIGN_H
#define FACE_ALIGN_H
// Face alignment library using dlib. 

#include<string>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

#define DEFAULT_RESIZE_DIM 192

namespace face_rec_lib {

    static cv::Rect dlibRectangleToOpenCV(dlib::rectangle r)
    {
        if (r.is_empty())
            return cv::Rect();
        return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
    }

    static dlib::rectangle openCVRectToDlib(cv::Rect r)
    {
        if (r.area() <=0 )
            return dlib::rectangle();
        return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
    }

    // Some heuristic vars
    typedef struct FACE_ALIGN_TRANS_VAR {
        float rotation;        // Rotation angle
        float pitching; // Pitching rate (good in raise face). Use the change of mid_eye_to_top_lip/whole_face_height as the approximate estimate of the face pitching rate.
        float ar_change; // Change of face aspect ratio (fair in down face, good in face fragment).
        float lr_change; // Change of width_of_left_face / width_of_right_face
    } FaceAlignTransVar;

    class FaceAlign
    {
        public:
            FaceAlign(const std::string & face_predictor);
            ~FaceAlign();

            // Detect face using dlib.
            std::vector<dlib::rectangle> getAllFaceBoundingBoxes(dlib::cv_image<dlib::bgr_pixel> & rgb_img);
            std::vector<cv::Rect> getAllFaceBoundingBoxes(cv::Mat & rgb_img);
            dlib::rectangle getLargestFaceBoundingBox(dlib::cv_image<dlib::bgr_pixel> & rgb_img);
            cv::Rect getLargestFaceBoundingBox(cv::Mat & rgb_img);
            dlib::full_object_detection getLargestFaceLandmarks(dlib::cv_image<dlib::bgr_pixel> & rgb_img);
            std::vector<cv::Point2f> getLargestFaceLandmarks(cv::Mat & rgb_img);
            std::vector<dlib::full_object_detection> getAllFaceLandmarks(dlib::cv_image<dlib::bgr_pixel> & rgb_img);
            std::vector<std::vector<cv::Point2f> > getAllFaceLandmarks(cv::Mat & rgb_img);
            // Find face landmarks.
            std::vector<dlib::point> findLandmarks(dlib::cv_image<dlib::bgr_pixel> &rgb_img, dlib::rectangle bb);
            // Do affine transform to align face.
            cv::Mat align(dlib::cv_image<dlib::bgr_pixel> &rgb_img,
                    dlib::rectangle bb=dlib::rectangle(),
                    const int img_dim=DEFAULT_RESIZE_DIM,
                    const int landmark_indices[]=FaceAlign::INNER_EYES_AND_TOP_LIP,
                    const float scale_factor=0.0);
            cv::Mat  align(cv::Mat & rgb_img,
                    cv::Rect rect=cv::Rect(),
                    const int img_dim=DEFAULT_RESIZE_DIM,
                    const int landmark_indices[]=FaceAlign::INNER_EYES_AND_TOP_LIP,
                    const float scale_factor=0.0);
            cv::Mat align(dlib::cv_image<dlib::bgr_pixel> &rgb_img,
                    cv::Mat & H,  // The affine matrix to the template
                    cv::Mat & inv_H, // Inverse affine matrix
                    dlib::rectangle bb=dlib::rectangle(),
                    const int img_dim=DEFAULT_RESIZE_DIM,
                    const int landmark_indices[]=FaceAlign::INNER_EYES_AND_TOP_LIP,
                    const float scale_factor=0.0);
            cv::Mat align(cv::Mat &rgb_img,
                    cv::Mat & H,  // The affine matrix to the template
                    cv::Mat & inv_H, // Inverse affine matrix
                    cv::Rect rect=cv::Rect(),
                    const int img_dim=DEFAULT_RESIZE_DIM,
                    const int landmark_indices[]=FaceAlign::INNER_EYES_AND_TOP_LIP,
                    const float scale_factor=0.0);
            cv::Mat  align(dlib::cv_image<dlib::bgr_pixel> &rgb_img,
                    cv::Mat & H, // The affine matrix to the template
                    FaceAlignTransVar & V,  // Transform variables in face alignment
                    dlib::rectangle bb=dlib::rectangle(),
                    const int img_dim=DEFAULT_RESIZE_DIM,
                    const int landmark_indices[]=FaceAlign::INNER_EYES_AND_TOP_LIP,
                    const float scale_factor=0.0);
            cv::Mat  align(cv::Mat &rgb_img,
                    cv::Mat & H, // The affine matrix to the template
                    FaceAlignTransVar & V, // Transform variables in face alignment
                    cv::Rect rect=cv::Rect(),
                    const int img_dim=DEFAULT_RESIZE_DIM,
                    const int landmark_indices[]=FaceAlign::INNER_EYES_AND_TOP_LIP,
                    const float scale_factor=0.0);

            // Align of lightened-cnn, mxnet version.
            cv::Rect  detectAlignCropLightenedCNNMxnet(cv::Mat & cv_img,
                    cv::Mat & face,
                    const int crop_size=128,
                    const float ts=0.1,
                    cv::Rect rect=cv::Rect(),
                    const int landmark_indices[]=FaceAlign::INNER_EYES_AND_BOTTOM_LIP
                    );
            // Align of lightened-cnn, author original version.
            cv::Rect  detectAlignCropLightenedCNNOrigin(cv::Mat & cv_img,
                    cv::Mat & face,
                    const int crop_size=128,
                    const int ec_mc_y = 48,
                    const int ec_y = 40,
                    cv::Rect rect=cv::Rect(),
                    const int landmark_indices[]=FaceAlign::OUTER_EYES_AND_NOSE_AND_MOUTH_CORNERS
                    );

            // Detect the largest face, align and crop it.
            cv::Mat detectAlignCrop(const cv::Mat &img,
                    cv::Rect & rect,
                    const int img_dim=DEFAULT_RESIZE_DIM,
                    const int landmark_indices[]=FaceAlign::INNER_EYES_AND_TOP_LIP,
                    const float scale_factor=0.0);
            cv::Mat detectAlignCrop(const cv::Mat &img,
                    cv::Rect & rect,
                    cv::Mat & H,  // The affine matrix to the template
                    cv::Mat & inv_H, // Inverse affine matrix
                    const int img_dim=DEFAULT_RESIZE_DIM,
                    const int landmark_indices[]=FaceAlign::INNER_EYES_AND_TOP_LIP,
                    const float scale_factor=0.0);

            // Detect face(s);
            void detectFace(const cv::Mat & img, std::vector<cv::Rect> & rects);
            void detectFace(const cv::Mat & img, cv::Rect & rect);

            // Landmark indices corresponding to the inner eyes and top lip.
            static const int INNER_EYES_AND_TOP_LIP[];
            // Landmark indices corresponding to the inner eyes and bottom lip.
            static const int INNER_EYES_AND_BOTTOM_LIP[];
            // Landmark indices corresponding to the inner eyes and nose.
            static const int OUTER_EYES_AND_NOSE[];

            // Five point landmark indices corresponding to the outer eyes, nose, and mouth corners.
            static const int OUTER_EYES_AND_NOSE_AND_MOUTH_CORNERS[];
            // Five point landmark indices corresponding to the inner eyes, nose, and mouth corners.
            static const int INNER_EYES_AND_NOSE_AND_MOUTH_CORNERS[];
            // Five point landmark indices corresponding to the outer eyes, nose, and top/bottom lips.
            static const int OUTER_EYES_AND_NOSE_AND_LIPS[];
            // Five point landmark indices corresponding to the inner eyes, nose, and and top/bottom lips.
            static const int INNER_EYES_AND_NOSE_AND_LIPS[];

        private:
            // Face landmark template data
            static float TEMPLATE_DATA[][2];
            // Face landmark template
            cv::Mat TEMPLATE;
            // Column normalized face landmark template
            cv::Mat MINMAX_TEMPLATE;

            dlib::frontal_face_detector detector;
            dlib::shape_predictor predictor;
    };
}
#endif // FACE_ALIGN_H
