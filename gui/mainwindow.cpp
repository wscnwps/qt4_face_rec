#include "gui/mainwindow.h"
//#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    _display = new MainWidget(this);
    setCentralWidget(_display);
}

MainWindow::~MainWindow()
{
}

