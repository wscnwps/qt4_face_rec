#ifndef FACE_FEATURE_H
#define FACE_FEATURE_H
#include <stdio.h>
#include "opencv2/opencv.hpp"

namespace face_rec_lib {
    using namespace cv;
    using namespace std;

    class FaceFeature {
        public:
            FaceFeature(const string &model_file, 
                    const string &trained_file,
                    const string & feature_name, 
                    const string & mean_file = string(), 
                    bool use_gpu = false);
            ~FaceFeature();

            void ExtractFaceFeature(const Mat &face, Mat &feature);
            void ExtractFaceNormFeature(const Mat &face, Mat &feature);
            Rect DetectFaceAndExtractFaceFeature(const Mat &img, Mat &feature);
            Rect DetectFaceAndExtractFaceNormFeature(const Mat &img, Mat &feature);
            float Compare(const Mat & img1, const Mat & img2);
            float CompareFeature(const Mat & f1, const Mat & f2);
            Rect DetectFaceAndAlign(const Mat & img, Mat & face);
            int GetFeatureDim();

        private:
            void *  _classifier;
            cv::Size _input_size;
            int _input_channels;
            int _feature_dim;
            string _feature_name;
    };
}

#endif //FACE_FEATURE_H
