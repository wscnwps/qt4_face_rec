#include <dlib/assert.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

#include "face_align.h"

//#include "opencv2/opencv.hpp"
//using namespace cv;
//using namespace std;

#ifndef M_PI
define M_PI  3.14159265358979323846
#endif

namespace face_rec_lib {
// Three-point landmarks.
const int FaceAlign::INNER_EYES_AND_BOTTOM_LIP[] =  {3, 39, 42, 57};
const int FaceAlign::OUTER_EYES_AND_NOSE[] = {3, 36, 45, 33};
const int FaceAlign::INNER_EYES_AND_TOP_LIP[] = {3, 39, 42, 51};

// Five-point landmarks.
const int FaceAlign::OUTER_EYES_AND_NOSE_AND_MOUTH_CORNERS[] =  {5, 36, 45, 33, 48, 54};
const int FaceAlign::INNER_EYES_AND_NOSE_AND_MOUTH_CORNERS[] =  {5, 39, 42, 33, 48, 54};
const int FaceAlign::OUTER_EYES_AND_NOSE_AND_LIPS[] =  {5, 36, 45, 33, 51, 57};
const int FaceAlign::INNER_EYES_AND_NOSE_AND_LIPS[] =  {5, 39, 42, 33, 51, 57};

float FaceAlign::TEMPLATE_DATA[][2] =
{
    {0.0792396913815, 0.339223741112}, {0.0829219487236, 0.456955367943},
    {0.0967927109165, 0.575648016728}, {0.122141515615, 0.691921601066},
    {0.168687863544, 0.800341263616}, {0.239789390707, 0.895732504778},
    {0.325662452515, 0.977068762493}, {0.422318282013, 1.04329000149},
    {0.531777802068, 1.06080371126}, {0.641296298053, 1.03981924107},
    {0.738105872266, 0.972268833998}, {0.824444363295, 0.889624082279},
    {0.894792677532, 0.792494155836}, {0.939395486253, 0.681546643421},
    {0.96111933829, 0.562238253072}, {0.970579841181, 0.441758925744},
    {0.971193274221, 0.322118743967}, {0.163846223133, 0.249151738053},
    {0.21780354657, 0.204255863861}, {0.291299351124, 0.192367318323},
    {0.367460241458, 0.203582210627}, {0.4392945113, 0.233135599851},
    {0.586445962425, 0.228141644834}, {0.660152671635, 0.195923841854},
    {0.737466449096, 0.182360984545}, {0.813236546239, 0.192828009114},
    {0.8707571886, 0.235293377042}, {0.51534533827, 0.31863546193},
    {0.516221448289, 0.396200446263}, {0.517118861835, 0.473797687758},
    {0.51816430343, 0.553157797772}, {0.433701156035, 0.604054457668},
    {0.475501237769, 0.62076344024}, {0.520712933176, 0.634268222208},
    {0.565874114041, 0.618796581487}, {0.607054002672, 0.60157671656},
    {0.252418718401, 0.331052263829}, {0.298663015648, 0.302646354002},
    {0.355749724218, 0.303020650651}, {0.403718978315, 0.33867711083},
    {0.352507175597, 0.349987615384}, {0.296791759886, 0.350478978225},
    {0.631326076346, 0.334136672344}, {0.679073381078, 0.29645404267},
    {0.73597236153, 0.294721285802}, {0.782865376271, 0.321305281656},
    {0.740312274764, 0.341849376713}, {0.68499850091, 0.343734332172},
    {0.353167761422, 0.746189164237}, {0.414587777921, 0.719053835073},
    {0.477677654595, 0.706835892494}, {0.522732900812, 0.717092275768},
    {0.569832064287, 0.705414478982}, {0.635195811927, 0.71565572516},
    {0.69951672331, 0.739419187253}, {0.639447159575, 0.805236879972},
    {0.576410514055, 0.835436670169}, {0.525398405766, 0.841706377792},
    {0.47641545769, 0.837505914975}, {0.41379548902, 0.810045601727},
    {0.380084785646, 0.749979603086}, {0.477955996282, 0.74513234612},
    {0.523389793327, 0.748924302636}, {0.571057789237, 0.74332894691},
    {0.672409137852, 0.744177032192}, {0.572539621444, 0.776609286626},
    {0.5240106503, 0.783370783245}, {0.477561227414, 0.778476346951}
};

FaceAlign::FaceAlign(const std::string & face_predictor)
{
    detector = dlib::get_frontal_face_detector();
    dlib::deserialize(face_predictor) >> predictor;

    TEMPLATE = cv::Mat(68, 2, CV_32FC1, TEMPLATE_DATA);
    TEMPLATE.copyTo(MINMAX_TEMPLATE);
    // Column normalize template to (0,1). It will make the face landmarks tightly inside (0,1) box.
    cv::Mat temp, cp;
    for (int i=0; i<TEMPLATE.cols; i++)
    {
        cv::normalize(TEMPLATE.col(i), temp, 1, 0, cv::NORM_MINMAX);
        cp = MINMAX_TEMPLATE.colRange(i, i+1);
        temp.copyTo(cp);
    }
}

FaceAlign::~FaceAlign()
{
}

std::vector<dlib::rectangle> FaceAlign::getAllFaceBoundingBoxes(dlib::cv_image<dlib::bgr_pixel> & rgb_img)
{
    return detector(rgb_img);
}

std::vector<cv::Rect> FaceAlign::getAllFaceBoundingBoxes(cv::Mat & rgb_img) {
  dlib::cv_image<dlib::bgr_pixel> cimg(rgb_img);
  std::vector<dlib::rectangle> bb = getAllFaceBoundingBoxes(cimg);
  std::vector<cv::Rect> res;
  for (int i = 0; i < bb.size(); i++) 
    res.push_back(dlibRectangleToOpenCV(bb[i]));
  return res;
}

dlib::rectangle FaceAlign::getLargestFaceBoundingBox(dlib::cv_image<dlib::bgr_pixel> & rgb_img)
{
    std::vector<dlib::rectangle> dets = this->getAllFaceBoundingBoxes(rgb_img);
    if (dets.size() > 0)
    {
        int i_max = 0;
        float max = 0;
        for (int i=0; i< dets.size(); i++)
        {
            float rect = dets[i].width() * dets[i].height();
            if( rect > max)
            {
                max = rect;
                i_max = i;
            }
        }
        return dets[i_max];
    }
    else
    {
        return dlib::rectangle();
    }
}

cv::Rect FaceAlign::getLargestFaceBoundingBox(cv::Mat & rgb_img) {
  dlib::cv_image<dlib::bgr_pixel> cimg(rgb_img);
  return dlibRectangleToOpenCV(getLargestFaceBoundingBox(cimg));
}

dlib::full_object_detection FaceAlign::getLargestFaceLandmarks(dlib::cv_image<dlib::bgr_pixel> & rgb_img) {
  dlib::rectangle bb = this->getLargestFaceBoundingBox(rgb_img);
  dlib::full_object_detection landmarks = this->predictor(rgb_img, bb);
  return landmarks;
}

std::vector<cv::Point2f> FaceAlign::getLargestFaceLandmarks(cv::Mat & rgb_img) {
  dlib::cv_image<dlib::bgr_pixel> cimg(rgb_img);
  dlib::full_object_detection landmarks = getLargestFaceLandmarks(cimg);
  std::vector<cv::Point2f> res;
  if (landmarks.num_parts() <= 0)
    return res;
  for (int i = 0; i < 68; i++) {
    cv::Point2f p(landmarks.part(i).x(), landmarks.part(i).y());
    res.push_back(p);
  }
  return res;
}

std::vector<dlib::full_object_detection> FaceAlign::getAllFaceLandmarks(dlib::cv_image<dlib::bgr_pixel> & rgb_img) {
  std::vector<dlib::full_object_detection> res;
  std::vector<dlib::rectangle> dets = getAllFaceBoundingBoxes(rgb_img);
  for (int i = 0; i < dets.size(); i++) 
    res.push_back(this->predictor(rgb_img, dets[i]));
  return res;
}

std::vector<std::vector<cv::Point2f> > FaceAlign::getAllFaceLandmarks(cv::Mat & rgb_img){
  dlib::cv_image<dlib::bgr_pixel> cimg(rgb_img);
  std::vector<std::vector<cv::Point2f> > res;
  std::vector<dlib::full_object_detection> dets = getAllFaceLandmarks(cimg);
  for (int d = 0 ; d < dets.size(); d++) {
    std::vector<cv::Point2f> lm;
    for (int i = 0; i < 68; i++) {
      cv::Point2f p(dets[d].part(i).x(), dets[d].part(i).y());
      lm.push_back(p);
    }
    res.push_back(lm);
  }
  return res;
}

cv::Mat  FaceAlign::align(dlib::cv_image<dlib::bgr_pixel> &rgb_img,
                          dlib::rectangle bb,
                          const int img_dim,
                          const int landmark_indices[],
                          const float scale_factor)
{
    cv::Mat H, inv_H;
    return align(rgb_img, H, inv_H, bb, img_dim, landmark_indices, scale_factor);
}

cv::Mat  FaceAlign::align(cv::Mat & rgb_img,
                          cv::Rect rect,
                          const int img_dim,
                          const int landmark_indices[],
                          const float scale_factor)
{
    cv::Mat H, inv_H;
    dlib::cv_image<dlib::bgr_pixel> cimg(rgb_img);
    return align(cimg, H, inv_H, openCVRectToDlib(rect), img_dim, landmark_indices, scale_factor);
}

cv::Mat  FaceAlign::align(cv::Mat &rgb_img,
                          cv::Mat & H,
                          cv::Mat & inv_H,
                          cv::Rect rect,
                          const int img_dim,
                          const int landmark_indices[],
                          const float scale_factor)
{
    dlib::cv_image<dlib::bgr_pixel> cimg(rgb_img);
    return align(cimg, H, inv_H, openCVRectToDlib(rect), img_dim, landmark_indices, scale_factor);
}

cv::Mat  FaceAlign::align(cv::Mat &rgb_img,
                          cv::Mat & H,
                          FaceAlignTransVar & V,
                          cv::Rect rect,
                          const int img_dim,
                          const int landmark_indices[],
                          const float scale_factor)
{
    dlib::cv_image<dlib::bgr_pixel> cimg(rgb_img);
    return align(cimg, H, V, openCVRectToDlib(rect), img_dim, landmark_indices, scale_factor);
}

cv::Mat  FaceAlign::align(dlib::cv_image<dlib::bgr_pixel> &rgb_img,
                          cv::Mat & H,
                          FaceAlignTransVar & V,
                          dlib::rectangle bb,
                          const int img_dim,
                          const int landmark_indices[],
                          const float scale_factor)
{
    if (bb.is_empty())
        bb = this->getLargestFaceBoundingBox(rgb_img);

    dlib::full_object_detection landmarks = this->predictor(rgb_img, bb);

    int nPoints = landmark_indices[0];
	cv::Point2f * src_points = new cv::Point2f[nPoints];
	cv::Point2f * dst_points = new cv::Point2f[nPoints];
	cv::Point2f * tp_points = new cv::Point2f[nPoints];

    cv::Mat template_face = TEMPLATE;
    if (scale_factor > 0 && scale_factor < 1) {
        template_face = MINMAX_TEMPLATE;
    }

    for (int i=1; i<=nPoints; i++)
    {
        dlib::point p = landmarks.part(landmark_indices[i]);
        src_points[i-1] = cv::Point2f(p.x(), p.y());
        dst_points[i-1] = cv::Point2f((float)img_dim * template_face.at<float>(landmark_indices[i], 0),
                                     (float)img_dim * template_face.at<float>(landmark_indices[i], 1));
        tp_points[i-1] = cv::Point2f(template_face.at<float>(landmark_indices[i], 0),
                                     template_face.at<float>(landmark_indices[i], 1));
        //std::cout<<dst_points[i-1]<<std::endl;
    }
    // Point in the middle of the eyes.
    cv::Point2f mid_eye_tp( (tp_points[0].x+tp_points[1].x)/2, (tp_points[0].y+tp_points[1].y)/2 );
    cv::Point2f mid_eye_src( (src_points[0].x+src_points[1].x)/2, (src_points[0].y+src_points[1].y)/2 );

    float resize_factor = 1.0;
    if (scale_factor > 0 && scale_factor < 1) {
       /*
        // The first two landmarks (inner/outer eyes) and the third landmark (bottom/top lip/nose) form an isosceles triangle approximately.
        float d1, d2, d3, h1, h2, h;
        d1 = cv::norm(dst_points[0] - dst_points[1]);
        d2 = cv::norm(dst_points[2] - dst_points[0]);
        d3 = cv::norm(dst_points[2] - dst_points[1]);
        h1 = std::sqrt(d2*d2 - d1*d1/4); // Height computed by landmark 0, 2
        h2 = std::sqrt(d3*d3 - d1*d1/4); // Height computed by landmark 1, 3
        h = (h1 + h2)/2; // Use their average
        resize_factor = scale_factor/ h * img_dim;
        //std::cout<<" "<<d1<<" "<<d2<<" "<<d3<<" "<<h1<<" "<<h2<<" "<<h<<" "<<resize_factor<<std::endl;
        */
        float h = cv::norm(tp_points[2] - mid_eye_tp);
        resize_factor = scale_factor/ h;
    }
    for (int i=0; i<nPoints; i++)
    {
        dst_points[i] -= cv::Point2f(0.5*img_dim, 0.5*img_dim);
        dst_points[i] *= resize_factor;
        dst_points[i] += cv::Point2f(0.5*img_dim, 0.5*img_dim);
        //std::cout<<dst_points[i]<<std::endl;
    }
    //Get affine matrix.
    H = cv::getAffineTransform(src_points, dst_points);

    //-------------------------------------------------------
    // Transform variables.
    // Use line connecting point in the middle of eyes and the top/bottom nose/lip point as the vertical axis.
    //float delta_y = template_face.at<float>(51, 1) - template_face.at<float>(27, 1);
    //float delta_x = template_face.at<float>(51, 0) - template_face.at<float>(27, 0);
    float delta_y = tp_points[2].y - mid_eye_tp.y;
    float delta_x = tp_points[2].x - mid_eye_tp.x;
    float angle_temp = atan2(delta_y, delta_x);
    //std::cout<<"arctan(y/x) "<<atan2(delta_y, delta_x)<<std::endl;
    //delta_y = landmarks.part(51).y() - landmarks.part(27).y();
    //delta_x = landmarks.part(51).x() - landmarks.part(27).x();
    delta_y = src_points[2].y - mid_eye_src.y;
    delta_x = src_points[2].x - mid_eye_src.x;
    float angle_img = atan2(delta_y, delta_x);
    float angle_rotation = (angle_img - angle_temp) * 180 / dlib::pi;
    //std::cout<<"arctan(y/x) "<<-1*atan2(delta_y, delta_x)<<std::endl;

    cv::Mat rotImg = dlib::toMat(rgb_img);
    cv::Mat H_rot = cv::getRotationMatrix2D(cv::Point(rotImg.cols/2, rotImg.rows/2), angle_rotation, 1);

    float value_tp[4] = {1e10, 1e10, -1e10, -1e10}; // min{x}, min{y}, max{x}, max{y} of the template
    float value_rot[4] = {1e10, 1e10, -1e10, -1e10}; // min{x}, min{y}, max{x}, max{y} of the rotated image
    std::vector<cv::Point> rotPoints;
    for ( int i = 0; i < template_face.rows; i++ ) {
      dlib::point p = landmarks.part(i);
      cv::Point result;
      result.x = H_rot.at<double>(0, 0)*p.x() +
                 H_rot.at<double>(0, 1)*p.y() +
                 H_rot.at<double>(0, 2);
      result.y = H_rot.at<double>(1, 0)*p.x() +
                 H_rot.at<double>(1, 1)*p.y() +
                 H_rot.at<double>(1, 2);
      rotPoints.push_back(result);

      cv::Point pt(template_face.at<float>(i, 0), template_face.at<float>(i, 1));
      value_tp[0] = value_tp[0] > pt.x ? pt.x : value_tp[0];
      value_tp[1] = value_tp[1] > pt.y ? pt.y : value_tp[1];
      value_tp[2] = value_tp[2] < pt.x ? pt.x : value_tp[2];
      value_tp[3] = value_tp[3] < pt.y ? pt.y : value_tp[3];

      value_rot[0] = value_rot[0] > result.x ? result.x : value_rot[0];
      value_rot[1] = value_rot[1] > result.y ? result.y : value_rot[1];
      value_rot[2] = value_rot[2] < result.x ? result.x : value_rot[2];
      value_rot[3] = value_rot[3] < result.y ? result.y : value_rot[3];
    }
    // Point in the middle of the eyes, after rotation.
    cv::Point2f mid_eye_rot( (rotPoints[landmark_indices[1]].x+rotPoints[landmark_indices[2]].x)/2,
        (rotPoints[landmark_indices[1]].y+rotPoints[landmark_indices[2]].y)/2 );

    //// Show rotated image for debug.
    //cv::line(rotImg, mid_eye_rot, rotPoints[landmark_indices[3]], CV_RGB(255, 0, 0));
    //cv::line(rotImg, cv::Point2f(value_rot[0],0), cv::Point2f(value_rot[0],300), CV_RGB(255, 0, 0));
    //cv::line(rotImg, cv::Point2f(value_rot[2],0), cv::Point2f(value_rot[2],300), CV_RGB(255, 0, 0));
    //cv::imshow("Roataed image", rotImg);
    //cv::waitKey(-1);

    // Aspect ratio of the temlate and the rotated image
    float ar_tp = (value_tp[2] - value_tp[0]) / (value_tp[3] - value_tp[1]);
    float ar_rot = (value_rot[2] - value_rot[0]) / (value_rot[3] - value_rot[1]);
    float ar_change = (ar_rot - ar_tp) / ar_tp;

    // Pitching rate. We use the change of mid_eye_to_top_lip/whole_face_height as the approximate estimate of the face pitching rate.
    float p_tp = (tp_points[2].y - mid_eye_tp.y) / (value_tp[3] - value_tp[1]);
    float p_rot = (rotPoints[landmark_indices[3]].y - mid_eye_rot.y) / (value_rot[3] - value_rot[1]);
    float pitching = (p_rot - p_tp) / p_tp;

    // Left and right face ratio of the template and the rotated image
    //// Use landmark 51 and 27's average x as the center vertial axis
    //float center_x_tp = (landmarks.part(51).x() + landmarks.part(27).x()) / 2;
    //float center_x_rot = (rotPoints[51].x + rotPoints[27].x) / 2;
    // Use line connecting mid_eye and bottom/top lip/nose as the vertical axis.
    float center_x_tp = (mid_eye_tp.x + tp_points[2].x) / 2;
    float center_x_rot = (mid_eye_rot.x + rotPoints[landmark_indices[3]].x) / 2;
    float lr_tp = (value_tp[2] - center_x_tp) / (center_x_tp - value_tp[0]);
    float lr_rot = (value_rot[2] - center_x_rot) / (center_x_rot - value_rot[0]);
    float lr_change = (lr_rot - lr_tp) / lr_tp;

//    std::cout<<"Rotation angle is : "<<angle_rotation<<std::endl;
//    std::cout<<"lr_change: "<<lr_tp<<" "<<lr_rot<<" "<<lr_change<<std::endl;
//    std::cout<<"ar_change: "<<ar_tp<<" "<<ar_rot<<" "<<ar_change<<std::endl;
//    std::cout<<"Pitching: "<<p_tp<<" "<<p_rot<<" "<<pitching<<std::endl;

    V.ar_change = ar_change;
    V.lr_change = lr_change;
    V.pitching = pitching;
    V.rotation = angle_rotation;
    //-------------------------------------------------------

    cv::Mat warpedImg = dlib::toMat(rgb_img);
    cv::warpAffine(warpedImg, warpedImg, H, cv::Size(img_dim, img_dim));
    
	delete[] src_points;
	delete[] dst_points;
	delete[] tp_points;
	return warpedImg;
}

cv::Mat  FaceAlign::align(dlib::cv_image<dlib::bgr_pixel> &rgb_img,
                          cv::Mat & H,
                          cv::Mat & inv_H,
                          dlib::rectangle bb,
                          const int img_dim,
                          const int landmark_indices[],
                          const float scale_factor)
{
    if (bb.is_empty())
        bb = this->getLargestFaceBoundingBox(rgb_img);

    dlib::full_object_detection landmarks = this->predictor(rgb_img, bb);

    int nPoints = landmark_indices[0];
	cv::Point2f * src_points = new cv::Point2f[nPoints];
	cv::Point2f * dst_points = new cv::Point2f[nPoints];
	cv::Point2f * tp_points = new cv::Point2f[nPoints];

    cv::Mat template_face = TEMPLATE;
    if (scale_factor > 0 && scale_factor < 1) {
        template_face = MINMAX_TEMPLATE;
    }

    for (int i=1; i<=nPoints; i++)
    {
        dlib::point p = landmarks.part(landmark_indices[i]);
        src_points[i-1] = cv::Point2f(p.x(), p.y());
        dst_points[i-1] = cv::Point2f((float)img_dim * template_face.at<float>(landmark_indices[i], 0),
                                     (float)img_dim * template_face.at<float>(landmark_indices[i], 1));
        tp_points[i-1] = cv::Point2f(template_face.at<float>(landmark_indices[i], 0),
                                     template_face.at<float>(landmark_indices[i], 1));
        //std::cout<<dst_points[i-1]<<std::endl;
    }
    // Point in the middle of the eyes.
    cv::Point2f mid_eye_tp( (tp_points[0].x+tp_points[1].x)/2, (tp_points[0].y+tp_points[1].y)/2 );
    cv::Point2f mid_eye_src( (src_points[0].x+src_points[1].x)/2, (src_points[0].y+src_points[1].y)/2 );
    
    float resize_factor = 1.0;
    if (scale_factor > 0 && scale_factor < 1) {
        float h = cv::norm(tp_points[2] - mid_eye_tp);
        resize_factor = scale_factor/ h;
    }
    for (int i=0; i<nPoints; i++)
    {
        dst_points[i] -= cv::Point2f(0.5*img_dim, 0.5*img_dim);
        dst_points[i] *= resize_factor;
        dst_points[i] += cv::Point2f(0.5*img_dim, 0.5*img_dim);
        //std::cout<<dst_points[i]<<std::endl;
    }
    //Get affine matrix.
    H = cv::getAffineTransform(src_points, dst_points);
    cv::invertAffineTransform(H, inv_H);
    //std::cout<<"invertAffineTransform: "<<inv_H<<std::endl;
    //inv_H = cv::getAffineTransform(dst_points, src_points);
    //std::cout<<"getAffineTransform: "<<inv_H<<std::endl;

    cv::Mat warpedImg = dlib::toMat(rgb_img);
    cv::warpAffine(warpedImg, warpedImg, H, cv::Size(img_dim, img_dim));

	delete [] src_points;
	delete[] dst_points;
	delete[] tp_points;

    return warpedImg;
}

// Align of lightened-cnn, mxnet version.
cv::Rect  FaceAlign::detectAlignCropLightenedCNNMxnet(cv::Mat & cv_img,
                                           cv::Mat & face,
                                           const int crop_size,
                                           const float ts,
                                           cv::Rect rect,
                                           const int landmark_indices[]
                                           ) {

    dlib::cv_image<dlib::bgr_pixel> rgb_img(cv_img);
    dlib::rectangle bb = openCVRectToDlib(rect);
    if (rect.area() <= 0)
    {
        bb = this->getLargestFaceBoundingBox(rgb_img);
        rect = dlibRectangleToOpenCV(bb);
    }
    if(rect.area() <= 0)
        return rect;

    dlib::full_object_detection landmarks = this->predictor(rgb_img, bb);

    int nPoints = landmark_indices[0];
    assert(3==nPoints);
    /*cv::Point2f src_points[nPoints];
    cv::Point2f dst_points[nPoints];
    cv::Point2f tp_points[nPoints];*/
	cv::Point2f * src_points = new cv::Point2f[nPoints];
	cv::Point2f * dst_points = new cv::Point2f[nPoints];
	cv::Point2f * tp_points = new cv::Point2f[nPoints];

    cv::Mat template_face = MINMAX_TEMPLATE;

    for (int i=1; i<=nPoints; i++)
    {
        dlib::point p = landmarks.part(landmark_indices[i]);
        src_points[i-1] = cv::Point2f(p.x(), p.y());
        dst_points[i-1] = cv::Point2f((float)crop_size * template_face.at<float>(landmark_indices[i], 0),
                (float)crop_size * template_face.at<float>(landmark_indices[i], 1));
        tp_points[i-1] = cv::Point2f(template_face.at<float>(landmark_indices[i], 0),
                template_face.at<float>(landmark_indices[i], 1));
        //std::cout<<dst_points[i-1]<<std::endl;
    }

    if (ts > 0 && ts < 1) {
        for (int i = 1; i <= nPoints - 1; i++) {
            dst_points[i-1] += cv::Point2f(0, crop_size*ts);
        }
        dst_points[nPoints-1] += cv::Point2f(0, crop_size*ts/2);
    }

    //// Point in the middle of the eyes.
    //cv::Point2f mid_eye_tp( (tp_points[0].x+tp_points[1].x)/2, (tp_points[0].y+tp_points[1].y)/2 );
    //cv::Point2f mid_eye_src( (src_points[0].x+src_points[1].x)/2, (src_points[0].y+src_points[1].y)/2 );
    //float resize_factor = 1.0;
    //if (scale_factor > 0 && scale_factor < 1) {
        //float h = cv::norm(tp_points[2] - mid_eye_tp);
        //resize_factor = scale_factor/ h;
    //}
    //for (int i=0; i<nPoints; i++)
    //{
        //dst_points[i] -= cv::Point2f(0.5*img_dim, 0.5*img_dim);
        //dst_points[i] *= resize_factor;
        //dst_points[i] += cv::Point2f(0.5*img_dim, 0.5*img_dim);
        ////std::cout<<dst_points[i]<<std::endl;
    //}

    //Get affine matrix.
    cv::Mat H = cv::getAffineTransform(src_points, dst_points);
    // Align and crop.
    cv::warpAffine(cv_img, face, H, cv::Size(crop_size, crop_size));

	delete [] src_points;
	delete[] dst_points;
	delete[] tp_points;
    return rect;
}

// Align of lightened-cnn, author original version.
cv::Rect  FaceAlign::detectAlignCropLightenedCNNOrigin(cv::Mat & cv_img,
                                                       cv::Mat & face,
                                                       const int crop_size,
                                                       const int ec_mc_y,
                                                       const int ec_y,
                                                       cv::Rect rect,
                                                       const int landmark_indices[]
                                                       ) {

    dlib::cv_image<dlib::bgr_pixel> rgb_img(cv_img);
    dlib::rectangle bb = openCVRectToDlib(rect);
    if (rect.area() <= 0) {
        bb = this->getLargestFaceBoundingBox(rgb_img);
        rect = dlibRectangleToOpenCV(bb);
    }
    if(rect.area() <= 0)
        return rect;

    dlib::full_object_detection landmarks = this->predictor(rgb_img, bb);

    int nPoints = landmark_indices[0];
    assert(5==nPoints);
    /*cv::Point2f src_points[nPoints];
    cv::Point2f dst_points[nPoints];*/
	cv::Point2f * src_points = new cv::Point2f[nPoints];
	cv::Point2f * dst_points = new cv::Point2f[nPoints];

    cv::Mat template_face = MINMAX_TEMPLATE;

    for (int i=1; i<=nPoints; i++)
    {
        dlib::point p = landmarks.part(landmark_indices[i]);
        src_points[i-1] = cv::Point2f(p.x(), p.y());
        dst_points[i-1] = cv::Point2f((float)crop_size * template_face.at<float>(landmark_indices[i], 0),
                                     (float)crop_size * template_face.at<float>(landmark_indices[i], 1));
    }

    // Rotate input image accoring to the eye points.
    float ang_tan = (src_points[0].y - src_points[1].y) /(src_points[0].x - src_points[1].x) ;
    float ang = std::atan(ang_tan)/M_PI*180;
    cv::Mat im_warp, rot_matrix;
    cv::Point2f center(cv_img.cols/2.0, cv_img.rows/2.0);
    rot_matrix = cv::getRotationMatrix2D(center, ang, 1.0);
    cv::Rect rotate_bbox = cv::RotatedRect(center, cv::Size2f(cv_img.cols, cv_img.rows), ang).boundingRect();
    cv::warpAffine(cv_img, im_warp, rot_matrix, rotate_bbox.size());
    // Find new positions on rotated image.
    cv::Point2f eye_center, mouth_center;
    eye_center.x = (src_points[0].x + src_points[1].x) / 2;
    eye_center.y = (src_points[0].y + src_points[1].y) / 2;
    mouth_center.x = (src_points[3].x + src_points[4].x) / 2;
    mouth_center.y = (src_points[3].y + src_points[4].y) / 2;
    std::vector<cv::Point2f> vec;
    vec.push_back(eye_center);
    vec.push_back(mouth_center);
    cv::transform(vec, vec, rot_matrix);
    eye_center = vec[0];
    mouth_center = vec[1];
//    std::cout<<"eye_center after rotate: "<<eye_center<<"\n";

    // Resize
    float resize_scale = ec_mc_y/(mouth_center.y - eye_center.y);
    // Find new positions on resized image.
    eye_center.x =  std::round(eye_center.x * resize_scale);
    eye_center.y =  std::round(eye_center.y * resize_scale);
    // Do resize.
    cv::resize(im_warp, im_warp,
               cv::Size(round(im_warp.cols*resize_scale), round(im_warp.rows*resize_scale)),
               0, 0, CV_INTER_LINEAR);
//    std::cout<<"Resize factor: "<<resize_scale<<std::endl;
//    std::cout<<"eye_center after resize: "<<eye_center<<"\n";

    // Crop
    // Keep "ec_y" pixels from eye_center upwards.
    int crop_y = eye_center.y - ec_y;
    int crop_y_end = crop_y + crop_size - 1;
    // Let "eye_center" as the vertical axis of the face.
    int crop_x = eye_center.x - std::floor(crop_size/2.0);
    int crop_x_end = crop_x + crop_size - 1;
    int box[4] = {std::max(crop_x, 0), std::min(crop_x_end, im_warp.cols - 1),
                  std::max(crop_y, 0), std::min(crop_y_end, im_warp.rows - 1)};
//    std::cout<<"ROI: "<<box[0]<<" "<<box[1]<<" "<<box[2]<<" "<<box[3]<<std::endl;
    // Fill '0' to invalid pixels.
    face = cv::Mat::zeros(cv::Size(crop_size, crop_size), cv_img.type());
    // Copy face ROI to result Mat "face".
    cv::Mat face_in_image = im_warp(cv::Range(box[2], box[3]), cv::Range(box[0], box[1]));
    face_in_image.copyTo(face(cv::Range(box[2] - crop_y, box[3] - crop_y),
            cv::Range(box[0] - crop_x, box[1] - crop_x)));

	delete[] src_points;
	delete[] dst_points;

    return rect;
}

// Find and crop face by dlib.
cv::Mat FaceAlign::detectAlignCrop(const cv::Mat &img,
                                   cv::Rect & rect,
                                   cv::Mat & H,
                                   cv::Mat & inv_H,
                                   const int img_dim,
                                   const int landmark_indices[],
                                   const float scale_factor)
{
    // Detection
    /* Dlib detects a face larger than 80x80 pixels.
     * We can use pyramid_up to double the size of image,
     * then dlib can find faces in size of 40x40 pixels in the original image.*/
    // dlib::pyramid_up(img);
    dlib::cv_image<dlib::bgr_pixel> cimg(img);
    std::vector<dlib::rectangle> dets;
    dets.push_back(getLargestFaceBoundingBox(cimg)); // Use the largest detected face only
    if (0 == dets.size() || dets[0].is_empty())
    {
        rect = cv::Rect();
        return cv::Mat();
    }
    rect = dlibRectangleToOpenCV(dets[0]);

    // --Alignment
    return align(cimg, H, inv_H, dets[0], img_dim, landmark_indices, scale_factor);
}

cv::Mat FaceAlign::detectAlignCrop(const cv::Mat &img,
                        cv::Rect & rect,
                        const int img_dim,
                        const int landmark_indices[],
                        const float scale_factor)
{
    cv::Mat H, inv_H;
    return detectAlignCrop(img, rect, H, inv_H, img_dim, landmark_indices, scale_factor);
}

void FaceAlign::detectFace(const cv::Mat & img, cv::Rect & rect)
{
    dlib::cv_image<dlib::bgr_pixel> cimg(img);
    dlib::rectangle det = getLargestFaceBoundingBox(cimg);
    rect = dlibRectangleToOpenCV(det);
}

void FaceAlign::detectFace(const cv::Mat & img, std::vector<cv::Rect> & rects)
{
    dlib::cv_image<dlib::bgr_pixel> cimg(img);
    std::vector<dlib::rectangle> dets = getAllFaceBoundingBoxes(cimg);
    rects.clear();
    for (int i = 0; i < dets.size(); i++)
        rects.push_back(dlibRectangleToOpenCV(dets[i]));
}

}
