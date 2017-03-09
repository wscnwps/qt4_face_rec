#ifndef CLASSIFIER_H
#define CLASSIFIER_H
// Caffe classifier adapted from caffe c++ sample code. 

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

/* Pair (label, confidence) representing a prediction. */
// typedef std::pair<string, float> Prediction;
namespace face_rec_lib {
using namespace caffe; // NOLINT(build/namespaces)
using namespace std;
using namespace cv;

class Classifier {
public:
  Classifier(const string &model_file, const string &trained_file,
             const string &mean_file = string(), bool use_gpu = false);

  void ExtractLayerByName(const cv::Mat &img, const string &layer_name,
                             Mat &feature);
  int GetLayerDimByName(string & layer_name);
  cv::Size GetInputGeometry();
  int GetInputChannels();

private:
  void SetMean(cv::Scalar channel_mean =cv::Scalar(93.5940, 104.7624, 129.1863)); // set mean by value
  void SetMean(const string &mean_file);
  void WrapInputLayer(std::vector<cv::Mat> *input_channels);
  void Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels);

private:
  std::shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};
}
#endif // CLASSIFIER_H
