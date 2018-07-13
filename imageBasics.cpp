#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "OpencvInclude.h"
#include "move_det.h"
using namespace cv;
using namespace std;

#define USE_CAMERA 0
VideoCapture cap;
char ukey[20]="C1414526A0CC8E9ED80";
int main( int argc, char** argv )
{   Mat image;
    cv::Size InImage_size(640,480);
#if USE_CAMERA 
    cap.open(-1);
    cap>>image;
     resize(image, image, InImage_size);
#else
      cap.open("/home/exbot/SLAM/Check_Background/build/5.avi");//<<----5-------------------------------------------------------------------------------
      cap>>image;
     resize(image, image, InImage_size);
#endif
    // 判断图像文件是否正确读取
    if ( image.data == nullptr ) //数据不存在,可能是文件不存在
    {
        cerr<<"文件"<<argv[1]<<"不存在."<<endl;
        return 0;
    }

	Move_det move_detector;

	move_detector.init_detector(10, 88, 0.7, 25, 1, 0, ukey);
	for(;;)
	{       
	       Mat capIn;
		cap >> capIn;	
		if(! capIn.data)break;
		resize(capIn, capIn, InImage_size);
		move_detector.update(capIn);
		for (int i=0;i<move_detector.move_targets.size();i++)
		    rectangle (capIn,   move_detector.move_targets[i],  Scalar(0, 255, 255),2,1,0);
		imshow("s",capIn);
		 if(waitKey(10) >= 0) break;
	}

}
