CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(qt4_face_rec)
set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/install)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)

add_subdirectory(face_rec_lib)

#QT-------------------------------------------------------------START
FIND_PACKAGE(Qt4 REQUIRED)
INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})  # This line is to include QT intermediate head files.
#QT source files
SET(qt_HEADERS gui/mainwindow.h gui/mainwidget.h  gui/imageviewer.h src/camera.h src/face_processor.h)
#SET(qt_FORMS gui/mainwidget.ui)
#SET(qt_RESOURCES face_demo.qrc)
QT4_WRAP_CPP(qt_HEADERS_MOC ${qt_HEADERS})
QT4_WRAP_UI(qt_FORMS_HEADERS ${qt_FORMS})
QT4_ADD_RESOURCES(qt_RESOURCES_RCC ${qt_RESOURCES})
#QT-------------------------------------------------------------END

find_package(dlib REQUIRED)
include_directories(${dlib_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Boost REQUIRED)
find_package(Boost COMPONENTS filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

#set(FaceRecLib_LIBRARIES "face_rec_lib;caffe")
set(FaceRecLib_LIBRARIES "face_rec_lib")

SET(all_SOURCES   src/main.cpp
    src/camera.cpp
    src/face_processor.cpp
    src/face_repo_utils.h
    src/face_repo_utils.cpp
    src/settings.h
    gui/mainwindow.cpp
    gui/mainwidget.cpp
    gui/imageviewer.cpp )

# target
ADD_EXECUTABLE(face_demo ${all_SOURCES} ${qt_HEADERS_MOC} ${qt_FORMS})
TARGET_LINK_LIBRARIES(face_demo ${FaceRecLib_LIBRARIES}  ${OpenCV_LIBRARIES}  ${Boost_LIBRARIES} ${QT_LIBRARIES} ${dlib_LIBRARIES})

