#include "face_processor.h"

#include <QCoreApplication>
#include <QDateTime>
#include <exception>

#include "face_rec_lib/face_feature.h"
#include "face_rec_lib/face_repository.h"
#include "face_rec_lib/face_align.h"
#include "face_repo_utils.h"
#include "settings.h"

namespace fs = ::boost::filesystem;
using namespace cv;
using namespace std;
using namespace face_rec_lib;

FaceProcessor::FaceProcessor(QObject *parent, bool processAll) : QObject(parent), _process_all(processAll)
{
    string appPath = QCoreApplication::applicationDirPath().toStdString();
    _caffe_prototxt = fs::path(appPath + "/" + caffe_prototxt).string();
    _caffe_model = fs::path(appPath + "/" + caffe_model).string();
    if (caffe_mean.length() > 0)
        _caffe_mean =  fs::path(appPath + "/" + caffe_mean).string();
    else
        _caffe_mean = "";
    _caffe_feature_layer_name = caffe_feature_layer_name;
    _dlib_face_model_path = fs::path(appPath + "/" + dlib_face_model_path).string();
    _face_repo_path = fs::path(appPath + "/" + face_repo_path).string();
    _face_image_home = fs::path(appPath + "/" + face_image_home).string();

    // Init
    _feature_extractor = new FaceFeature(
        _caffe_prototxt,
        _caffe_model,
        _caffe_feature_layer_name,
        _caffe_mean);
    _face_align = new FaceAlign(_dlib_face_model_path);
    _work_state = STATE_DEFAULT;

    Q_ASSERT(_feature_extractor != NULL);
    Q_ASSERT(_face_align != NULL);

    // Load face repository.
    _face_repo = new FaceRepo(*_feature_extractor);
    faceRepoInit();

    _face_repo_is_dirty = false;
    _save_timer.start(FACE_REPO_TIME_INTERVAL_TO_SAVE*1000, this);

    // Dispatch faces by person.
    if ( 0 < _face_image_path.size() )
        dispatchImage2Person();
    /*// Print result to test dispatchImage2Person()
    cout<<"Person count: "<<person_.size()<<endl;
    for (int i = 0; i < person_.size(); i++)
    {
        cout<<person_[i]<<": "<<person_image_path_[i].size()<<endl;
        for (int j = 0; j < _person_image_path[i].size(); j++)
            cout<<j<<": "<<person_image_path_[i][j]<<endl;
    }*/

    // Pre allocate an empty (white) result image.
    _empty_result = Mat(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT, CV_8UC3, Scalar(255, 255, 255));

    // Face recognition parameter
    _face_rec_knn = FACE_REC_KNN;
    _face_rec_th_dist = FACE_REC_TH_DIST;
    _face_rec_th_num = FACE_REC_TH_N;
    // Face verification parameter
    _face_ver_th_dist = FACE_VER_TH_DIST;
    _face_ver_num = FACE_VER_NUM;
    _face_ver_valid_num = FACE_VER_VALID_NUM;
    _face_ver_sample_num = FACE_VER_SAMPLE_NUM;
    // Person register parameter
    _face_reg_num = FACE_REG_NUM;
    _face_reg_need_ver = false;
    // Face pose selection parameter
    _selected_faces_num = 0;
    _selected_face_ver_valid_num = 0;
    _feature_min_dist = FEATURE_MIN_DIST;
    _feature_max_dist = FEATURE_MAX_DIST;
}

FaceProcessor::~FaceProcessor()
{
    if (NULL != _face_repo)
    {
        if (_face_repo_is_dirty)
        {
            qDebug()<<"Updating face repository and save before quit, please wait.";
            _face_repo->Save(_face_repo_path);
        }
        delete _face_repo;
    }
    delete _face_align;
    delete _feature_extractor;
}

bool FaceProcessor::faceRepoInit()
{
    bool suc_load;
    try
    {   // Load face repository.
        suc_load = _face_repo->Load(_face_repo_path);
        int N = _face_repo->GetFaceNum();
        for (int i = 0; i < N; i++) {
          _face_image_path.push_back(_face_repo->GetPath(i));
        }
    }
    catch(exception e)
    {
        qDebug()<<"Face repository does not exist or other error.";
        qDebug()<<e.what();
        suc_load = false;
        _face_image_path.clear();
    }

    if (suc_load)
        return true;

    QDir dir;
    dir.mkpath(QString::fromLocal8Bit(_face_repo_path.c_str()));

    // Try reconstruct face repository from images.
    vector<fs::path> image_path;
    getAllFiles(fs::path(_face_image_home), ".jpg", image_path);
    _face_image_path.clear();
    if ( image_path.size() > 0)
    {
        for ( int i = 0; i < image_path.size(); i++ )
            _face_image_path.push_back(image_path[i].string());
        qDebug()<<"Try to construct face repository from images.";
        _face_repo->InitialIndex(_face_image_path);
        _face_repo->Save(_face_repo_path);
        return true;
    }
    return false;
}

// Dispatch FaceRepo's image paths to each person.
void FaceProcessor::dispatchImage2Person()
{
    for (int i =  0 ; i < _face_image_path.size(); i++)
    {
        string person_name = getPersonName(_face_image_path[i]);
        vector<string>::iterator iter = find(_person.begin(), _person.end(), person_name);
        if ( iter == _person.end() )
        {
            _person.push_back(person_name);
            vector<string> person_images;
            person_images.push_back(_face_image_path[i]);
            _person_image_path.push_back(person_images);
        }
        else
        {
            int pos = iter - _person.begin();
            _person_image_path[pos].push_back(_face_image_path[i]);
        }
    }
}

void FaceProcessor::slotProcessFrame(const cv::Mat &frame)
{
    if (_process_all)
        process(frame);
    else
        queue(frame);
}

void FaceProcessor::setProcessAll(bool all)
{
    _process_all = all;
}

void FaceProcessor::process(cv::Mat frame)
{
    // Results for show widgets
    QList<cv::Mat> result_faces;
    QStringList result_names;
    QList<float> result_sim;

    // Detect and align face.
    Mat face_aligned, H, inv_H;
    Rect rect_face_detected;
#ifdef LIGHTENED_CNN
    rect_face_detected = _face_align->detectAlignCropLightenedCNNOrigin(frame, face_aligned);
#else
    face_aligned = _face_align->detectAlignCrop(frame, rect_face_detected, H, inv_H,
                                                FACE_ALIGN_SCALE,
                                                FACE_ALIGN_POINTS,
                                                FACE_ALIGN_SCALE_FACTOR);
#endif
//    if (H.size().area() > 0 )
//    {
//        cout<<"------------------"<<endl;
//        cout<<H<<endl;
//        cout<<inv_H<<endl;
//        cout<<affine2square(H)*affine2square(inv_H)<<endl;
//    }

    // No face detected.
    if (0 == rect_face_detected.area())
    {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        emit sigDisplayImageReady(frame);
        emit sigCaptionUpdate("Cannot detect face");
        return;
    }

    // Prepare main camera view.
//    cout<<"FaceProcessor::process: detected face rect:"<<rect_face_detected<<endl;
    cv::rectangle(frame, rect_face_detected, cv::Scalar( 255, 0, 255 ));
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    emit sigDisplayImageReady(frame);

    Mat feature;
    _feature_extractor->ExtractFaceNormFeature(face_aligned, feature);
//    feature = _face_repo->GetFeatureCV(1); // For test

    // Main process
    switch (_work_state)
    {
    case STATE_DEFAULT: // Face recognition
    {
        map<float, pair<int, string> > combined_result; // Combine recognition result by using FaceRepo.
        map <string, string> example_face; // Example face for each group.
        faceRecognition( feature, SIMPLE_MIN(_face_rec_knn, _face_repo->GetValidFaceNum()), _face_rec_th_dist, combined_result, example_face);
        cout<<"Face recognition return group num: "<<combined_result.size()<<endl;
        for (map<float, pair<int, string> >::iterator it = combined_result.begin();
             it != combined_result.end(); it++)
            cout<<"Group \""<<(it->second).second<<"\": num "<<(it->second).first<<", ave_dist "<<it->first<<endl;

        // Prepare results to show.
        map<float, pair<int, string> >::iterator it = combined_result.begin();
        for (int i = 0; i < RESULT_FACES_NUM; i++)
        {
            if ( i < combined_result.size() && (it->second).first > _face_rec_th_num)
            {
                string name = (it->second).second;
                Mat face_in_repo = imread(example_face[name]);
//                Mat face_in_repo = frame;
//                cout<<example_face[name]<<endl;
                resize(face_in_repo, face_in_repo, Size(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT));
                cv::cvtColor(face_in_repo, face_in_repo, cv::COLOR_BGR2RGB);
                result_faces.append(face_in_repo);
                result_names.append(QString::fromStdString(name));
                result_sim.append(dist2sim(it->first));
                it++;
            }
            else
            {
                result_faces.append(_empty_result); // Use pre-allocated empty image.
//                // For test
//                if( i < 3)
//                {
//                    cv::Mat result;
//                    cv::resize(frame, result, cv::Size(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT));
//                    result_faces.append(result);
//                }
//                else
//                    result_faces.append(empty_result_);
                result_names.append("");
                result_sim.append(0);
            }
        }
//        cout<<"In process: "<<int(result_faces[0].data[99])<<endl;
        emit sigResultFacesReady(result_faces, result_names, result_sim);
        emit sigCaptionUpdate(QString("Face Recognition"));
        break;
    }
    case STATE_VERIFICATION: // Person verification
    {
        // Verification done.
        if ( _selected_faces_num >= _face_ver_num )
            break;
        // The first iteration of face verification.
        if ( 0 == _selected_faces_num )
        {
            vector<string>::iterator iter = find(_person.begin(), _person.end(), _face_reg_ver_name);
            if ( iter != _person.end() ) // Found the name in the face repository.
            { // Show a sample face registered in face repo.
                string sample_path = _person_image_path[iter-_person.begin()][0];
                _face_ver_target_sample = imread(sample_path);
                resize(_face_ver_target_sample, _face_ver_target_sample, Size(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT));
                cv::cvtColor(_face_ver_target_sample, _face_ver_target_sample, cv::COLOR_BGR2RGB);
            }
            else
            { // The name is not in the face repository.
                if ( !_face_ver_target_sample.empty() )
                    _face_ver_target_sample.release();
                emit sigCaptionUpdate("The specified name does not exist.");
                emit sigVerificationDone();
                break;
            }
        }

        // Check face pose of the current frame.
        if (!checkFacePose(feature, H, inv_H)) {
            emit sigCaptionUpdate("Please shake the head around.");
            break;
        }

        // Add current face to the face verification stack.
        verAndSelectFace(face_aligned, feature, H, inv_H);

        // Prepare results to show.
        for (int i = 0; i < RESULT_FACES_NUM-1; i++)
        {
            if ( i < _selected_faces_num )
            {
                Mat face;
                resize(_selected_face_aligned[i], face, Size(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT));
                cv::cvtColor(face, face, cv::COLOR_BGR2RGB);
                result_faces.append(face);
                QString result = _selected_face_ver_valid[i] ? "Matched" : "Unmatched";
                result_names.append(result);
                result_sim.append(0);
            }
            else
            {
                result_faces.append(_empty_result); // Use pre-allocated empty image.
                result_names.append("");
                result_sim.append(0);
            }
        }
        // Use the last result to show a sample of the target person.
        result_faces.append(_face_ver_target_sample);
        result_names.append("Ver target");
        result_sim.append(0);
        QString caption = "Face Verification";
        // Face verification stack full. Make decision.
        if ( _selected_faces_num == _face_ver_num)
        {
            if ( _selected_face_ver_valid_num >= _face_ver_valid_num )
                caption = "Face Verfication: ACCEPT!";
            else
                caption = "Face Verfication: DENY!";
            emit sigVerificationDone();
        }
        emit sigResultFacesReady(result_faces, result_names, result_sim);
        emit sigCaptionUpdate(caption);
        break;
    }
    case STATE_REGISTER:  // Person register.
    {
        // Register done
        if ( _selected_faces_num >= SIMPLE_MAX(_face_reg_num, _face_ver_num) || // Register successly done.
             _face_reg_need_ver && _selected_faces_num >= _face_ver_num && _selected_face_ver_valid_num < _face_ver_valid_num ) // Register failed.
            break;
        // The first iteration of face register.
        if ( 0 == _selected_faces_num )
        {
            vector<string>::iterator iter = find(_person.begin(), _person.end(), _face_reg_ver_name);
            if ( iter != _person.end() ) // Person already in the face repository.
            {
                _face_reg_need_ver = true;
                string sample_path = _person_image_path[iter-_person.begin()][0];
                _face_ver_target_sample = imread(sample_path);
                resize(_face_ver_target_sample, _face_ver_target_sample, Size(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT));
                cv::cvtColor(_face_ver_target_sample, _face_ver_target_sample, cv::COLOR_BGR2RGB);
            }
        }
        // Check face pose of the current frame.
        if (!checkFacePose(feature, H, inv_H)) {
            emit sigCaptionUpdate("Please shake head around.");
            break;
        }

        // Add current face to the face verification stack.
        verAndSelectFace(face_aligned, feature, H, inv_H);
        QString caption = "Face Register";
        // The person already exist, and man in front of the camera has not passed the verification.
        if ( _face_reg_need_ver &&
             _selected_faces_num >= _face_ver_num &&
             _selected_face_ver_valid_num < _face_ver_valid_num )
        {
                caption = "Face Register: DENY due to verfication failure!";
                emit sigVerificationDone();
        }
        else if ( _selected_faces_num == _face_reg_num )
        {// Do register.
            vector<string> filelist;
            fs::path save_dir(_face_image_home);
            save_dir /= fs::path(_face_reg_ver_name);
            fs::create_directories(save_dir);
            for (int i = 0; i< _selected_faces_num; i++)
            {
                fs::path filepath = save_dir;
                filepath /= fs::path((QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss_")
                                      + QString::number(i)).toStdString() + ".jpg");
                imwrite(filepath.string(), _selected_face_aligned[i]);
                filelist.push_back(filepath.string());
                _face_image_path.push_back(filepath.string());
            }
            if (_face_repo->GetFaceNum() == 0) // First person adds to the face repository.
                faceRepoInit();
            else {
                _face_repo->AddFace(filelist, _selected_face_feature);
                _face_repo_is_dirty = true;
            }
            _person_image_path.push_back(filelist);
            _person.push_back(_face_reg_ver_name);
            caption = "Face Register: SUCCESS!";
            emit sigRegisterDone();
        }
        // Prepare results to show.
        for (int i = 0; i < RESULT_FACES_NUM; i++)
        {
            if ( i < _selected_faces_num )
            {
                Mat face;
                resize(_selected_face_aligned[i], face, Size(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT));
                cv::cvtColor(face, face, cv::COLOR_BGR2RGB);
                result_faces.append(face);
                result_names.append("");
                result_sim.append(0);
            }
            else
            {
                result_faces.append(_empty_result); // Use pre-allocated empty image.
                result_names.append("");
                result_sim.append(0);
            }
        }
        if (_face_reg_need_ver)
        {
            result_faces.last() = _face_ver_target_sample;
            result_names.last() = "Existed target";
        }
        emit sigResultFacesReady(result_faces, result_names, result_sim);
        emit sigCaptionUpdate(caption);
        break;
    }
    }
}

void FaceProcessor::faceRecognition( const Mat & query,  const int knn, const float th_dist,
                                     map<float, pair<int, string> > & combined_result,
                                     map <string, string> & example_face)
{
    if ( 0 == _face_repo->GetValidFaceNum())
        return;

    vector<string> return_list;
    vector<int> return_list_pos;
    vector<float>  dists;

//    cout<<"FaceProcessor::faceRecognition VALID FACE NUM IN REPOSITORY: "<<face_repo_->GetValidFaceNum()<<endl;
    _face_repo->Query(query, knn, return_list, return_list_pos, dists);
//    cout<<"FaceProcessor::faceRecognition return_list.size "<<return_list.size()<<endl;

    // Group return faces by person.
    vector <string> person_name;
    vector <int> person_count;
    vector <float> person_dist;
    for (int j = 0 ; j < knn; j++)
    {
        if (dists[j] > th_dist)
            continue;
        string class_name = getPersonName(return_list[j]);
        example_face.insert(pair<string, string>(class_name, return_list[j]));
        vector<string>::iterator iter = find(person_name.begin(), person_name.end(), class_name);
        if ( iter == person_name.end() )
        {
            person_name.push_back(class_name);
            person_dist.push_back(dists[j]);
            person_count.push_back(1);
        }
        else
        {
            int pos = iter - person_name.begin();
            person_dist[pos] += dists[j];
            person_count[pos] ++;
        }
    }
    // Sort groups by average dist. (std::map will sort according to the first item automatically)
    for (int j = 0; j < person_name.size(); j++)
    {
        person_dist[j] /= person_count[j];
        combined_result.insert(pair<float, pair<int, string> >
                               (person_dist[j],  pair<int, string>(person_count[j], person_name[j]))  );
    }
}

// TODO
// Now only use feature distance to check face pose.
// We'd better use face pose directly.
bool FaceProcessor::checkFacePose(const Mat & feature, const Mat & H, const Mat & inv_H)
{
    // TODO
    // Use front face only by checking H and inv_H.
    if (false)
        return false;

    if ( 0 == _selected_faces_num )
        return true;
    for ( int i = 0; i < _selected_faces_num; i++ )
    {
        Mat f = _selected_face_feature[i];
        if (f.size() != feature.size())
            f = f.t();
        double dist = norm(feature, f);

        qDebug()<<"FaceProcessor::checkFacePose()"<<" "<<dist;
        if ( dist > _feature_max_dist ) // Maybe another person.
            return false;
        if ( dist < _feature_min_dist) // Ignore similar face.
            return false;
    }
    return true;
}

void FaceProcessor::verAndSelectFace(const Mat & face, const Mat & feature, const Mat & H, const Mat & inv_H)
{
    // Verificate the face with the given person name.
    bool match = false;

    if (STATE_VERIFICATION  == _work_state ||
            STATE_REGISTER == _work_state && _face_reg_need_ver) {
        srand(time(0));
        // Select and compare samples from face repository.
        vector<string>::iterator iter = find(_person.begin(), _person.end(), _face_reg_ver_name);
        int pos_person = iter - _person.begin();
        // Try "_face_ver_sample_num" times to verificate a face.
        // Each time compare the input "face" with a sample random select from the face repo.
        // The input "face" is accept if it matched with any sample.
        for (int i = 0; !match && i < SIMPLE_MIN(_face_ver_sample_num, _person_image_path[pos_person].size()); i ++)
        {
            int j = rand() % _person_image_path[pos_person].size();
            Mat f = _face_repo->GetFeatureCV(_person_image_path[pos_person][j]);
            if (f.size() != feature.size())
                f = f.t(); // transport matrix
            match = match || norm(f, feature) < _face_ver_th_dist;
            cout<<"Verification, dist to "<<_person_image_path[pos_person][j]<<": "<<norm(f, feature)<<endl;
        }
    }

    // Add face into the list.
    _selected_face_aligned.push_back(face);
    _selected_face_feature.push_back(feature);
    _selected_face_H.push_back(H);
    _selected_face_inv_H.push_back(inv_H);
    _selected_face_ver_valid.push_back(match);
    _selected_face_ver_valid_num += match ? 1 : 0;
    _selected_faces_num ++;
}

void FaceProcessor::cleanSelectedFaces()
{
    _selected_face_aligned.clear();
    _selected_face_feature.clear();
    _selected_face_H.clear();
    _selected_face_inv_H.clear();
    _selected_face_ver_valid.clear();
    _selected_faces_num = 0;
    _selected_face_ver_valid_num = 0;
    _face_reg_need_ver = false;
}

void FaceProcessor::timerEvent(QTimerEvent *ev)
{
    // Timer to lock the camera frame for processing.
    if (ev->timerId() == _frame_timer.timerId())
    {
        process(_frame);
//        qDebug()<<"FaceProcessor::timerEvent() frame released."<<endl;
        _frame.release();
        _frame_timer.stop(); // Open queue
        return;
    }

    // Timer to save face repository.
    if(ev->timerId() == _save_timer.timerId() && NULL != _face_repo)
    {
        _save_timer.stop();
        if (_face_repo_is_dirty)
        {
            _face_repo->Save(_face_repo_path);
            _face_repo_is_dirty = false;
        }
        _save_timer.start(FACE_REPO_TIME_INTERVAL_TO_SAVE*1000, this);
    }
}

void FaceProcessor::queue(const cv::Mat &frame)
{
//    if (!frame.empty())
//        qDebug() << "FaceProcessor::queue() Converter dropped frame !";

    _frame = frame;
    // Lock current frame by timer.
    if (!_frame_timer.isActive())
        _frame_timer.start(0, this);

    // If "_frame_timer.isActive()", the input "frame" is dropped.
}

void FaceProcessor::slotRegister(bool start, QString name)
{
    cleanSelectedFaces();
    if (start)
        _work_state = STATE_REGISTER;
    else
        _work_state = STATE_DEFAULT;
    if (!name.isEmpty())
        _face_reg_ver_name = name.toStdString();
}

void FaceProcessor::slotVerification(bool start, QString name)
{
    cleanSelectedFaces();
    if (start)
        _work_state = STATE_VERIFICATION;
    else
        _work_state = STATE_DEFAULT;
    if (!name.isEmpty())
        _face_reg_ver_name = name.toStdString();

    cout<<"FaceProcessor::slotVerification:  "<<_face_reg_ver_name<<endl;
}



