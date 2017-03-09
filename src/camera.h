#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <QObject>
#include <QScopedPointer>
#include <QTimerEvent>
#include <stdio.h>
#include <iostream>
#include <QImage>
#include <QBasicTimer>
#include <QDebug>

class Camera : public QObject
{
    Q_OBJECT
    QScopedPointer<cv::VideoCapture> _video_capture;
    QBasicTimer _timer;
    bool _run;
    bool _using_video_camera;
    int _camera_index;
    cv::String _video_file_name;

public:
    Camera(int camera_index=0, QObject* parent=0) : QObject(parent)
    {
        _camera_index = camera_index;
        _using_video_camera = true;
    }

    ~Camera();
    QImage convertToQImage( cv::Mat frame );


public slots:
    void slotRun();
    void slotCameraIndex(int index);
    void slotVideoFileName(QString fileName);
    void slotUsingVideoCamera(bool value);
    void slotStopped();

signals:
    void sigStarted();
    void sigMatReady(const cv::Mat &);

private:
    void timerEvent(QTimerEvent * ev);
};

#endif
