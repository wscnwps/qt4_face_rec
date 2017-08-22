#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>

#include "opencv2/opencv.hpp"
#include "boost/filesystem.hpp"
#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "face_repository.h"
#include "face_feature.h"
#include "face_align.h"

#define LIGHTENED_CNN

#define FACE_ALIGN_SIZE 224
//#define FACE_ALIGN_POINTS FaceAlign::INNER_EYES_AND_BOTTOM_LIP
//#define FACE_ALIGN_SCALE_FACTOR 0.3
#define FACE_ALIGN_POINTS FaceAlign::INNER_EYES_AND_TOP_LIP
#define FACE_ALIGN_SCALE_FACTOR 0.26 // 25:100 as in CASIA dataset paper. Use 96*96 as input in small CNN, that's 25/96.

namespace fs = ::boost::filesystem;

using namespace cv;
using namespace std;
using namespace face_rec_lib;

void getAllFiles(const fs::path &root, const string &ext,
                 vector<fs::path> &ret) {
    if (!fs::exists(root) || !fs::is_directory(root))
        return;

    fs::recursive_directory_iterator it(root);
    fs::recursive_directory_iterator endit;

    while (it != endit) {
        if (fs::is_regular_file(*it) && it->path().extension() == ext)
            ret.push_back(it->path());
        ++it;
    }
}

int main(int argc, char **argv) {
    string root = "../../";
    // Path to model files
    string caffe_prototxt = root + "./models/cnn/LightenedCNN_B_deploy.prototxt";
    string caffe_model = root + "./models/cnn/LightenedCNN_B.caffemodel";
    string caffe_feature_layer_name = "eltwise_fc1";
    string align_model = root + "./models/dlib_shape_predictor_68_face_landmarks.dat";
    // Path to image root.
    string image_root(root + "./images");
    string ext(".jpg");
    // Initial face feature extractor and face aligner
    FaceFeature feature_extractor(caffe_prototxt, caffe_model,
            caffe_feature_layer_name);
    FaceAlign face_align(align_model);

    vector<fs::path> file_path;
    getAllFiles(fs::path(image_root), ext, file_path);

    int N = file_path.size();
    if (0 == N)
    {
        cerr<<"No image found in the given path with \""<<ext<<"\" extension."<<endl;
        exit(-1);
    }

    // Detect face, then save to the disk.
    cout<<"Image(s):"<<endl;
    for (int i = 0; i < N; i++) {
        cout<<i+1<<"/"<<N<<": "<<file_path[i]<<endl;
        Mat face = imread(file_path[i].string());
        Mat face_cropped;
        Rect face_detect;
#ifndef LIGHTENED_CNN
        Mat H, inv_H;
        face_cropped = face_align.detectAlignCrop(face, face_detect, H, inv_H, FACE_ALIGN_SCALE, FACE_ALIGN_POINTS, FACE_ALIGN_SCALE_FACTOR );
#else
        face_detect = face_align.detectAlignCropLightenedCNNOrigin(face, face_cropped); 
#endif
        if (0 == face_detect.area()) {
            cerr<<"No face detected"<<endl;
            fs::remove(file_path[i]);
            continue;
        }

        // Save to the disk.
        imwrite(file_path[i].string(), face_cropped);
    }

    return 0;
}
