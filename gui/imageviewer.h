#ifndef IMAGEVIEWER_H
#define IMAGEVIEWER_H

#include <QWidget>
#include <QImage>
#include <QPainter>
#include <QDebug>
#include <opencv2/opencv.hpp>

class ImageViewer : public QWidget
{
    Q_OBJECT
public:
    explicit ImageViewer( QWidget *parent = 0, int handle = -1 );
    void paintEvent(QPaintEvent *);

public slots:
    void slotSetImage(const cv::Mat & frame, int handle = -1);

private:
    cv::Mat _frame;
    QImage _image;
    int _handle;
};

#endif // IMAGEVIEWER_H
