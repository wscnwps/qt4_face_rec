#ifndef SETTINGS_H
#define SETTINGS_H
// Better to use a config file instead of this setting file.

#include <string>

// For GUI
// Result face size
#define RESULT_FACE_WIDTH 80
#define RESULT_FACE_HEIGHT 80
// Result face count
#define RESULT_FACES_NUM 6

// For FaceAlign
// Aligned face size
#define FACE_ALIGN_SCALE 224
// Feature points used to align the face
#define FACE_ALIGN_POINTS FaceAlign::INNER_EYES_AND_TOP_LIP
// Aligned face scale factor.
// It is the proportion of the distance, which from the center point of inner eyes to the top lip, in the whole image.
#define FACE_ALIGN_SCALE_FACTOR 0.26

// CNN Model file paths

//// Small net ------------------------------------------- START
//static const std::string caffe_prototxt = "../models/cnn/small.prototxt";
//static const std::string caffe_model = "../models/cnn/small.caffemodel";
//// Don't use mean file.
////static const std::string caffe_mean = "../models/cnn/small_mean_image.binaryproto";
////static const std::string caffe_mean = "../models/cnn/small_mean_117.binaryproto";
//static const std::string caffe_mean = "";
//static const std::string caffe_feature_layer_name = "prob";
//// Small net ------------------------------------------- END

// LightenedCNN_B net ------------------------------------------- START
static const std::string caffe_prototxt = "../models/cnn/LightenedCNN_B_deploy.prototxt";
static const std::string caffe_model = "../models/cnn/LightenedCNN_B.caffemodel";
static const std::string caffe_mean = "../models/cnn/LightenedCNN_B_mean.binaryproto";
//static const std::string caffe_mean = "";
static const std::string caffe_feature_layer_name = "eltwise_fc1";
#define LIGHTENED_CNN
// LightenedCNN_B net ------------------------------------------- END

// Face detection model path
static const std::string dlib_face_model_path = "../models/dlib_shape_predictor_68_face_landmarks.dat";
// Folder to save face repository
static const std::string face_repo_path = "../dataset";
static const std::string face_image_home = "../images";

// Parameters to control face pose in person verification or register.
// Face that has similar pose with the any accepted face will be discarded immediately.
// Minimum pose distance between faces in person verification or register.
#define POSE_MIN_DIST  0.2
// Use feature distance to select diverse input faces to ver or reg.
#define FEATURE_MIN_DIST 0.15
#define FEATURE_MAX_DIST 0.5

// Face recognition parameters.
#define FACE_REC_KNN  25  // Size of return (knn) list while searching face in face repository.
#define FACE_REC_TH_DIST  0.5 // Distance threshold for same person.
#define FACE_REC_TH_N  3 // Least number of samples with same label in the return list to recognize a face.

// Face verification parameters.
#define FACE_VER_TH_DIST 0.5  // Distance threshold for same person.
#define FACE_VER_SAMPLE_NUM  5 // Number of samples from the face repo to compare for the ver of a face.
#define FACE_VER_NUM  3 // Number of input faces to be checked in person verification.
#define FACE_VER_VALID_NUM  2 // Minimun number of accepted faces to verificate a person.

// Face register parameters.
#define FACE_REG_NUM (RESULT_FACES_NUM - 1) // Number of faces needed in person register.

// Time interval to save face repository (in sec.).
#define FACE_REPO_TIME_INTERVAL_TO_SAVE   3600

#endif // SETTINGS_H
