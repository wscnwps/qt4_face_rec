#include <stdio.h>
//#include <iterator>
//#include <algorithm>
//#include <glog/logging.h>
#include "opencv2/opencv.hpp"
#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "classifier.h"
//#include "face_align.h"
#include "face_feature.h"

//namespace fs = ::boost::filesystem;

namespace face_rec_lib {
    using namespace cv;
    using namespace std;

    FaceFeature::FaceFeature(const string &model_file, const string &trained_file, const string & feature_name, const string & mean_file, bool use_gpu)
        : _feature_name(feature_name)
    {
        _classifier = new Classifier(model_file, trained_file, mean_file, use_gpu);
        Classifier * classifier = (Classifier*) _classifier;

        _input_size = classifier->GetInputGeometry();
        _input_channels = classifier->GetInputChannels();
        _feature_dim = classifier->GetLayerDimByName(_feature_name);
        assert(_feature_dim > 0);
        assert(_input_channels == 1 || _input_channels == 3);
    }

    FaceFeature::~FaceFeature() {
        if ( NULL != _classifier ) {
            Classifier * classifier = (Classifier*) _classifier;
            delete classifier;
            _classifier = NULL;
        }
    }

    int FaceFeature::GetFeatureDim() {
        return _feature_dim;
    }

    void FaceFeature::ExtractFaceFeature(const Mat &face, Mat &feature){
        Classifier * classifier = (Classifier*) _classifier;
        Mat img(face);
        if (face.channels() == 3)
            cvtColor(face, img, CV_BGR2GRAY);
        img.convertTo(img, CV_32FC1, 1.0/255);
        classifier->ExtractLayerByName(img, _feature_name, feature);
    }

    void FaceFeature::ExtractFaceNormFeature(const Mat &face, Mat &feature){
        ExtractFaceFeature(face, feature);
        normalize(feature, feature, 1, 0, NORM_L2); 
    }

    Rect FaceFeature::DetectFaceAndExtractFaceFeature(const Mat &img, Mat &feature){
        Mat face;
        Rect face_rect = DetectFaceAndAlign(img, face);
        if (face_rect.area() > 0)
            ExtractFaceFeature(face, feature);
        return face_rect;
    }

    Rect FaceFeature::DetectFaceAndExtractFaceNormFeature(const Mat &img, Mat &feature){
        Rect face_rect = DetectFaceAndExtractFaceFeature(img, feature);
        normalize(feature, feature, 1, 0, NORM_L2); 
        return face_rect;
    }

    float FaceFeature::Compare(const Mat & img1, const Mat & img2){
        Mat f1, f2;
        ExtractFaceFeature(img1, f1);
        ExtractFaceFeature(img2, f2);
        return CompareFeature(f1, f2);
    }

    float FaceFeature::CompareFeature(const Mat & feature_a, const Mat & feature_b){
        double ab = feature_a.dot(feature_b);
        double aa = feature_a.dot(feature_a);
        double bb = feature_b.dot(feature_b);
        float sim = -ab / sqrt(aa*bb);
        return 0.5 + sim/2;
    }

    // Pseudo align. 
    Rect FaceFeature::DetectFaceAndAlign(const Mat & img, Mat & face){
        return Rect(0, 0, img.cols, img.rows);
    }
}
