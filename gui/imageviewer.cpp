#include "gui/imageviewer.h"

ImageViewer::ImageViewer( QWidget *parent, int handle ) : QWidget(parent)
{
    setAttribute(Qt::WA_OpaquePaintEvent);
    _handle = handle;
    _frame = cv::Mat();
    _image = QImage();
}

void ImageViewer::slotSetImage(const cv::Mat &frame, int handle)
{

    if (_handle != handle)
        return;

    if (!_image.isNull())
        qDebug() << "The last paint has not been finished. Viewer dropped frame!";

    this->_frame = frame.clone();
    if (_frame.size().height != size().width() || _frame.size().height != size().width())
        setFixedSize(_frame.size().width, _frame.size().height);
    update();
}

void ImageViewer::paintEvent(QPaintEvent *)
{
    // "data" in QImage is a ref. So we need to keep "_frame" alive.
    const QImage img((const unsigned char*)_frame.data, _frame.cols, _frame.rows, _frame.step,
                               QImage::Format_RGB888);
    _image = img;
    QPainter p(this);
    p.drawImage(0, 0, _image);
    _image = QImage();
}
