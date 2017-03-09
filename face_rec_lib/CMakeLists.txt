cmake_minimum_required (VERSION 2.8.11)
#project (face_recognition_library)

set(FaceRecLib_MAJOR_VERSION 0)
set(FaceRecLib_MINOR_VERSION 0)
set(FaceRecLib_PATCH_VERSION 1)
set(FaceRecLib_VERSION
  ${FaceRecLib_MAJOR_VERSION}.${FaceRecLib_MINOR_VERSION}.${FaceRecLib_PATCH_VERSION})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# ---[ Using cmake scripts and modules
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")
message("${CMAKE_MODULE_PATH}")

#if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) 
  #set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/install" CACHE PATH "Default install path" FORCE) 
#endif() 

# set up include-directories
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}"   # to find foo/foo.h
    "${CMAKE_CURRENT_BINARY_DIR}")  # to find foo/config.h

find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS}) # ex. -DCPU_ONLY
include_directories(include)
add_definitions(${Caffe_DEFINITIONS})    

find_package(dlib REQUIRED)
include_directories(${dlib_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Boost COMPONENTS filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
find_package(Flann REQUIRED)
include_directories(${FLANN_INCLUDE_DIRS})

configure_file(config.h.in "${CMAKE_CURRENT_BINARY_DIR}/config.h" @ONLY)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


#add_library (face_rec_lib STATIC 
add_library (face_rec_lib SHARED
  face_feature.h
  face_align.h 
  face_repository.h
  face_feature.cpp
  classifier.h
  classifier.cpp
  face_align.cpp
  face_repository.cpp
  config.h.in)
target_link_libraries(face_rec_lib ${dlib_LIBRARIES} ${Caffe_LIBRARIES})

add_executable(align_face tools/align_face_images.cpp)
target_link_libraries(align_face face_rec_lib ${dlib_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} )

set_target_properties(face_rec_lib PROPERTIES
  PUBLIC_HEADER "face_feature.h;face_repository.h;face_align.h;"
  VERSION ${FaceRecLib_VERSION})

#FRAMEWORK ON
#SOVERSION 0
install(TARGETS face_rec_lib 
  # IMPORTANT: Add the foo library to the "export-set"
  #EXPORT FaceRecLibTargets
  RUNTIME DESTINATION "bin" COMPONENT bin
  LIBRARY DESTINATION "lib" COMPONENT shlib
  PUBLIC_HEADER DESTINATION "include"
  COMPONENT dev)

install(FILES  cmake/Modules/FindFlann.cmake
  DESTINATION cmake/Modules)
get_filename_component(ABS_CAFFE_LIB libs/libcaffe.so REALPATH)
#get_filename_component(ABS_CAFFE_LIB ${Caffe_LIBRARIES} REALPATH)
install(FILES 
    ${ABS_CAFFE_LIB}
    DESTINATION lib)
get_filename_component(NAME_CAFFE_LIB ${ABS_CAFFE_LIB} NAME)
set( caffe_target ${NAME_CAFFE_LIB} )
set( caffe_link ${CMAKE_INSTALL_PREFIX}/lib/libcaffe.so )
install(CODE
    "execute_process(COMMAND cmake -E create_symlink ${caffe_target} ${caffe_link})")
