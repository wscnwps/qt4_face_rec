#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include <QWidget>
#include <QThread>
#include <QLabel>
#include <QStringList>
#include <QPushButton>
#include <QLineEdit>
#include <opencv2/opencv.hpp>

#include "gui/imageviewer.h"
#include "src/camera.h"
#include "src/face_processor.h"
#include "src/settings.h"

class MainWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MainWidget(QWidget *parent = 0);
    ~MainWidget();

signals:
    void sigResultFaceMatReady(const cv::Mat & frame, int handle);
    void sigRegister(bool start, QString name = QString());
    void sigVerification(bool start, QString name = QString());

public slots:
    void slotResultFacesReady(const QList<cv::Mat> result_faces,
                              const QStringList result_names,
                              const QList<float> result_sim);
    void slotCaptionUpdate(QString caption);
    void slotRegisterDone();
    void slotVerificationDone();

private slots:
    void slotOnClickRegisterButton();
    void slotOnClickVerificationButton();

private:
    FaceProcessor* _face_processor;
    Camera* _camera;
    QThread _face_process_thread;
    QThread _camera_thread;

    QLabel* _caption;

    // Camera view
    ImageViewer* _image_viewer;

    // Show result faces
    ImageViewer* _result_viewer[RESULT_FACES_NUM];
    QLabel* _result_rank[RESULT_FACES_NUM];
    QLabel* _result_name[RESULT_FACES_NUM];
    QLabel* _result_sim[RESULT_FACES_NUM];
    cv::Mat _result_faces[RESULT_FACES_NUM];

    // Control buttons
    QPushButton *_register_PushButton;
    QPushButton *_verification_PushButton;
    bool _is_reg; // Doing register.
    bool _is_ver; // Doing verification.
    QLineEdit* _name_LineEdit;
};

#endif // MAINWIDGET_H
