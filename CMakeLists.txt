cmake_minimum_required( VERSION 2.8 )
project( imageBasics )

# 添加c++ 11标准支持
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 寻找OpenCV库
find_package( OpenCV REQUIRED )
# 添加头文件
include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    "/usr/include/eigen3/"
    "./vie"
    "./kcf"
    "./include"
)
SET(_SRC imageBasics.cpp  move_det.cpp	  des.cpp  ./vie/VIBE.cpp
    ./vie/originalVibe.cpp    ./vie/select.cpp ./vie/main.cpp
    ./kcf/fhog.cpp  ./kcf/kcftracker.cpp)

SET(_SRC1 imageBasics.cpp )
    
add_executable( imageBasics  ${_SRC})
#add_executable( imageBasics  ${_SRC1})#lib test
# 链接OpenCV库

target_link_libraries( imageBasics
   ${OpenCV_LIBS}
  #/home/exbot/SLAM/Check_Background/build/libmove_detector.so
)

SET(LIB_SRC   move_det.cpp	 des.cpp ./vie/originalVibe.cpp    ./vie/select.cpp ./kcf/fhog.cpp  ./kcf/kcftracker.cpp)
ADD_LIBRARY(move_detector SHARED ${LIB_SRC})


