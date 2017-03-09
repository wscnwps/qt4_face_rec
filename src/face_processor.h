#ifndef FACE_PROCESSOR_H
#define FACE_PROCESSOR_H

#include <QObject>
#include <QBasicTimer>
#include <QTimerEvent>
#include <QDir>
#include <QDebug>
#include <QImage>
//#include <QString>
#include <QStringList>
#include <QResource>
//#include <QVector>
//#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include "face_rec_lib/face_repository.h"
#include "face_rec_lib/face_align.h"
#include "settings.h"

namespace fs = ::boost::filesystem;
using namespace cv;
using namespace std;
using namespace face_rec_lib;

// Declare metatypes to allow them transfer by QT sig and slot.
Q_DECLARE_METATYPE(cv::Mat)
Q_DECLARE_METATYPE(QList<cv::Mat>)
Q_DECLARE_METATYPE(QList<float>)

class FaceProcessor : public QObject
{
    Q_OBJECT

public:
    FaceProcessor(QObject *parent=0, bool processAll = false);
    ~FaceProcessor();

    // Process all images camera captured or not.
    void setProcessAll(bool all);

    // Get the class name of an image. JUST DEMO!
    // Now we assume that all face images of the same person store
    // in the same folder. So an face's person name is the fold name.
    string getPersonName(const string path) {
      return fs::canonical(path).parent_path().filename().string();
    }

    // Three work states. Default state is face recognition.
    enum PROCESS_STATE{STATE_DEFAULT = 0, STATE_VERIFICATION, STATE_REGISTER};

signals:
    void sigDisplayImageReady(const cv::Mat& frame);
    void sigResultFacesReady(const QList<cv::Mat> result_faces,
                             const QStringList result_names,
                             const QList<float> result_sim);
//    void sigNoFaceDetect();
    void sigCaptionUpdate(QString caption);
    void sigRegisterDone();
    void sigVerificationDone();

public slots:
    void slotProcessFrame(const cv::Mat& frame);
    void slotRegister(bool start, QString name);
    void slotVerification(bool start, QString name);

private:
    // To process camera capture images.
    void process(cv::Mat frame);
    void queue(const cv::Mat & frame);
    void timerEvent(QTimerEvent* ev);

    bool faceRepoInit();

    // Dispatch FaceRepo's image paths to each person.
    void dispatchImage2Person();

    // Do face recognition
    void faceRecognition( const cv::Mat & query, const int knn, const float th_dist,
                          map<float, pair<int, string> > & combined_result,
                          map <string, string> & example_face);

    // For face verification or register
    bool checkFacePose(const Mat & feature, const Mat & H, const Mat & inv_H); // Check if the current face is valid.
    void verAndSelectFace(const Mat & face, const Mat & feature, const Mat & H, const Mat & inv_H); // Select a face.
    void cleanSelectedFaces(); // Clean verification or register statement.
//    void faceRegister();


private:
    // Timer to save face repository.
    QBasicTimer _save_timer;
    bool _face_repo_is_dirty;

    // To process camera captured images.
    QBasicTimer _frame_timer;
    cv::Mat _frame;     // Current camera frame.
    bool _process_all;  // Process every frame captured by the camera?

    // Face detection, recognition, repository models' path.
    string _caffe_prototxt;
    string _caffe_model;
    string _caffe_mean;
    string _caffe_feature_layer_name;
    string _dlib_face_model_path;
    string _face_repo_path;
    string _face_image_home; // Root directory to store face images.

    // Face feature, alignment, repository classes.
    FaceFeature * _feature_extractor;
    FaceAlign * _face_align;
    FaceRepo * _face_repo;

    // Face repository.
    vector<string> _face_image_path;  // All image paths, used by "FaceRepo".
    vector <vector<string> > _person_image_path;  //Face image path for each person.
    vector <string> _person; // Person names.

    // Working statement: recognition, verification or register.
    PROCESS_STATE _work_state;

    // Empty (white) result images.
    cv::Mat _empty_result;

    // Face recognition parameters
    int  _face_rec_knn; // Size of return (knn) list while searching face in face repository.
    float _face_rec_th_dist; // Distance threshold for same person.
    int _face_rec_th_num; // Least number of samples with same label in the return list to recognize a face.

    // Face verification parameter
    float _face_ver_th_dist; // Distance threshold for same person.
    int _face_ver_sample_num; // Number of samples from the face repo to compare for the ver of a face.
    int _face_ver_num; // Number of faces to be checked in verification.
    int _face_ver_valid_num; // Minimun number of accepted faces to verificate a person.
    cv::Mat _face_ver_target_sample; // A sample face of verification target for shown.

    // Face register parameter
    int _face_reg_num; // Number of faces needed in register.
    string _face_reg_ver_name; // Person name in face register of verification.
    bool _face_reg_need_ver; // Need to verificate current person before register because the name already exist.

    // Select faces in person verification or register.
    // We should use faces with different poses for robust.
    int _selected_faces_num; // Number of selected faces.
    vector<cv::Mat> _selected_face_H;  // Affine matrix in alignment of selected faces.
    vector<cv::Mat> _selected_face_inv_H;  // Inverse affine matrix in alignment of selected faces.
    float  _pose_min_dist; // Minimum pose distance.
    vector<cv::Mat> _selected_face_aligned;  // Selected faces (aligned).
    vector<cv::Mat> _selected_face_feature;  // Feature of selected faces.
    vector<bool> _selected_face_ver_valid; // Is the selected face pass the verification.
    int _selected_face_ver_valid_num; // Number of selected face that passed the verification.
    float _feature_min_dist;  // Minimun feature distance to make different pose.
    float _feature_max_dist; // Maximum feature distance to assure same person in front of the camera.
};

#endif // FACE_PROCESSOR_H
