#include "gui/mainwindow.h"
#include "gui/mainwidget.h"
#include <QApplication>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType< QList<cv::Mat> >("QList<cv::Mat>");
    qRegisterMetaType< QList<float> >("QList<float>");

    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
