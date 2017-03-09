#include "camera.h"

Camera::~Camera()
{
}

void Camera::slotRun()
{
    if (!_video_capture || !_using_video_camera)
    {
        if (_using_video_camera)
            _video_capture.reset(new cv::VideoCapture(_camera_index));
        else
            _video_capture.reset(new cv::VideoCapture(_video_file_name));
    }
    if (_video_capture->isOpened())
    {
        _timer.start(0, this);
        emit sigStarted();
    }
}

void Camera::slotStopped()
{
    _timer.stop();
}

void Camera::timerEvent(QTimerEvent *ev)
{
    if (ev->timerId() != _timer.timerId())
        return;
    cv::Mat frame;
    if (!_video_capture->read(frame)) // Blocks until a new frame is ready
    {
        _timer.stop();
        return;
    }
    emit sigMatReady(frame);
}

void Camera::slotUsingVideoCamera(bool value)
{
    _using_video_camera = value;
}

void Camera::slotCameraIndex(int index)
{
    _camera_index = index;
}

void Camera::slotVideoFileName(QString fileName)
{
    _video_file_name = fileName.toStdString().c_str();
}
