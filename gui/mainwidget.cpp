#include "mainwidget.h"

//#include <iostream>
//#include <stdio.h>

#include <QVBoxLayout>
#include <QHBoxLayout>
//#include <QGridLayout>
//#include <QPushButton>
//#include <QRadioButton>
//#include <QLineEdit>
#include <QTimer>
#include <QRegExp>
#include <QValidator>
//#include <QComboBox>
//#include <QFileDialog>
#include <QDebug>

#include "gui/imageviewer.h"
#include "src/camera.h"
#include "src/face_processor.h"

MainWidget::MainWidget(QWidget *parent) :
    QWidget(parent)
{
    // Init state = recognition
    _is_reg = false;
    _is_ver = false;

    /*
     * GUI layout definition.
     */
    // Layout in the top for view
    QHBoxLayout * viewLayout = new QHBoxLayout();
    QVBoxLayout * cameraLayout = new QVBoxLayout();
    _caption = new QLabel("Face Recognition", this);
    cameraLayout->addWidget(_caption);
    // Camera view
    _image_viewer = new ImageViewer(this);
    cameraLayout->addWidget(_image_viewer);
    viewLayout->addLayout(cameraLayout);
    // Return face view
    QGridLayout * resultLayout = new QGridLayout();
    viewLayout->addLayout(resultLayout);
    for (int i = 0; i < RESULT_FACES_NUM; i++) {
        QVBoxLayout * oneResult = new QVBoxLayout();
        _result_rank[i] = new QLabel("#" + QString::number(i+1), this);
        _result_name[i] = new QLabel(this);
        _result_sim[i] = new QLabel(this);
        _result_viewer[i] = new ImageViewer(this, i);
        _result_viewer[i]->setFixedSize(RESULT_FACE_WIDTH, RESULT_FACE_HEIGHT);
        oneResult->addWidget(_result_rank[i], 0, Qt::AlignHCenter);
        oneResult->addWidget(_result_viewer[i], 0, Qt::AlignHCenter);
        oneResult->addWidget(_result_name[i]);
        oneResult->addWidget(_result_sim[i]);
        resultLayout->addLayout(oneResult, i/2, i%2);
    }

    // Layout in the bottom for input texts and push buttons.
    QHBoxLayout* controlLayout = new QHBoxLayout();
    // Input user name.
    QHBoxLayout* nameLayout = new QHBoxLayout();
    QLabel* nameLabel = new QLabel("Name or phone number (unique):", this);
    _name_LineEdit = new QLineEdit(this);
    QValidator *nameValidator = new QRegExpValidator(QRegExp("^[a-zA-Z0-9_-]+$"), this );
    _name_LineEdit->setValidator(nameValidator);
    nameLayout->addWidget(nameLabel);
    nameLayout->addWidget(_name_LineEdit);
    // Two control push bottons.
    _register_PushButton = new QPushButton("Register", this);
    _verification_PushButton = new QPushButton("Verification", this);
    // Add all to the controlLayout
    controlLayout->addLayout(nameLayout);
    controlLayout->addWidget(_register_PushButton);
    controlLayout->addWidget(_verification_PushButton);

    // Main layout
    QVBoxLayout * mainLayout = new QVBoxLayout(this);
    mainLayout->addLayout(viewLayout);
    mainLayout->addLayout(controlLayout);

    /*
     * Face recoginition and verification workflow.
     */
    _camera = new Camera();
    _face_processor = new FaceProcessor();
    _face_processor->setProcessAll(false);
    // Move camera and face detector to their independent threads
    _face_process_thread.start();
    _camera_thread.start();
    _camera->moveToThread(&_camera_thread);
    _face_processor->moveToThread(&_face_process_thread);
    // Make connection
    _image_viewer->connect(_face_processor,
                           SIGNAL(sigDisplayImageReady(cv::Mat)),
                           SLOT(slotSetImage(cv::Mat)));
    _face_processor->connect(_camera, SIGNAL(sigMatReady( cv::Mat)),
                           SLOT(slotProcessFrame(cv::Mat)));
    _face_processor->connect(this, SIGNAL(sigRegister(bool, QString)), SLOT(slotRegister(bool, QString)));
    _face_processor->connect(this, SIGNAL(sigVerification(bool, QString)), SLOT(slotVerification(bool, QString)));
    this->connect(_face_processor,
                  SIGNAL(sigCaptionUpdate(QString)),
                  SLOT(slotCaptionUpdate(QString)));
    this->connect(_face_processor,
                  SIGNAL(sigResultFacesReady(QList<cv::Mat>, QStringList,  QList<float>)),
                  SLOT(slotResultFacesReady(QList<cv::Mat>, QStringList, QList<float>)));
    this->connect(_face_processor, SIGNAL(sigRegisterDone()), SLOT(slotRegisterDone()));
    this->connect(_face_processor, SIGNAL(sigVerificationDone()), SLOT(slotVerificationDone()));
    this->connect(_register_PushButton, SIGNAL(clicked()), SLOT(slotOnClickRegisterButton()));
    this->connect(_verification_PushButton, SIGNAL(clicked()), SLOT(slotOnClickVerificationButton()));
    for (int i = 0; i < RESULT_FACES_NUM; i++) {
        _result_viewer[i]->connect(this, SIGNAL(sigResultFaceMatReady(cv::Mat, int)),
                                   SLOT(slotSetImage(cv::Mat, int )));
    }
    // Start camera.
    QTimer::singleShot(0, _camera, SLOT(slotRun()));
}

void MainWidget::slotCaptionUpdate(QString caption)
{
    _caption->setText(caption);
}

MainWidget::~MainWidget()
{
    _camera->disconnect();
    _face_processor->disconnect();
    _camera_thread.quit();
    _face_process_thread.quit();
    _camera_thread.wait();
    _face_process_thread.wait();
    _camera->~Camera();
    _face_processor->~FaceProcessor();
}

void MainWidget::slotResultFacesReady(const QList<cv::Mat> result_faces,
                          const QStringList result_names,
                          const QList<float> result_sim)
{
    for (int i = 0; i < RESULT_FACES_NUM; i++) {
        _result_name[i]->setText(result_names[i]);
        if ( 0 == result_sim[i])
            _result_sim[i]->setText("");
        else
            _result_sim[i]->setText(QString("%1").arg(result_sim[i]));
        _result_faces[i] = result_faces[i];
        emit sigResultFaceMatReady(_result_faces[i], i);
    }
}


void MainWidget::slotOnClickRegisterButton()
{
    if ( _is_reg )
    {
        _is_reg = false;
        _register_PushButton->setText("Register");
        _verification_PushButton->setEnabled(true);
        _name_LineEdit->setEnabled(true);
        emit sigRegister(false);
    }
    else
    {
        if (_name_LineEdit->text().isEmpty())
            return;
        _is_reg = true;
        _register_PushButton->setText("Reg Stop");
        _verification_PushButton->setEnabled(false);
        _name_LineEdit->setEnabled(false);
        emit sigRegister(true, _name_LineEdit->text());
    }
}

void MainWidget::slotOnClickVerificationButton()
{
    if ( _is_ver  )
    {
        _is_ver = false;
        _verification_PushButton->setText("Verification");
        _register_PushButton->setEnabled(true);
        _name_LineEdit->setEnabled(true);
        emit sigVerification(false);
    }
    else
    {
        if (_name_LineEdit->text().isEmpty())
            return;
        _is_ver = true;
        _verification_PushButton->setText("Ver Stop");
        _register_PushButton->setEnabled(false);
        _name_LineEdit->setEnabled(false);
        emit sigVerification(true, _name_LineEdit->text());
    }
}

void MainWidget::slotRegisterDone()
{
    _register_PushButton->setText("Continue");
}

void MainWidget::slotVerificationDone()
{
    _verification_PushButton->setText("Continue");
}
