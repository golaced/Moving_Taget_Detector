/*=================================================
 * Version:
 * v1.0: 原版程序由IplImage转换为Mat
===================================================
*/

#include "highgui.h"
#include "cv.h"
#include <iostream>
#include <select.h>
using namespace cv;
using namespace std;

float compute_absolute_mat2(const Mat& in, Mat & out,float thr)
{
        double average_flow=0;
	int cnt_flow_use;
	if (out.empty()){
		out.create(in.size(), CV_32FC1);
	}
 
	const Mat_<Vec2f> _in = in;
	//遍历吧，少年
	for (int i = 0; i < in.rows; ++i){
		float *data = out.ptr<float>(i);
		for (int j = 0; j < in.cols; ++j){
			double s = _in(i, j)[0] * _in(i, j)[0] + _in(i, j)[1] * _in(i, j)[1];
			if (s>thr){
				data[j] = std::sqrt(s);
				
			}
			else{
				data[j] = 0.0;
			}
			average_flow+=data[j];
		        cnt_flow_use++;
		}
	}
	
	return average_flow/cnt_flow_use;
}

int main1(int argc, char *argv[])
{
    VideoCapture capture;
    capture = VideoCapture("./1.avi");
  
    // 用于遍历capture中的帧，通道数为3，需要转化为单通道才可以处理
    Mat tmpFrame, tmpFrameF;
    // 当前帧，单通道，uchar / Float
    Mat currentFrame, currentFrameF;
    // 上一帧，单通道，uchar / Float
    Mat previousFrame, previousFrameF;

    int frameNum = 0;
     Mat gray,prvGray, optFlow ,absoluteFlow, img_for_show,and_image;
    capture >> tmpFrame;
    
    int rows = tmpFrame.rows;
    int cols = tmpFrame.cols;
    while(!tmpFrame.empty())
    {
        capture >> tmpFrame;
        //tmpFrame=cvQueryFrame(capture);
        frameNum++;
        if(frameNum == 1)
        {
            // 第一帧先初始化各个结构，为它们分配空间
            previousFrame.create(tmpFrame.size(), CV_8UC1);
            currentFrame.create(tmpFrame.size(), CV_8UC1);
            currentFrameF.create(tmpFrame.size(), CV_32FC1);
            previousFrameF.create(tmpFrame.size(), CV_32FC1);
            tmpFrameF.create(tmpFrame.size(), CV_32FC1);
        }

        if(frameNum >= 2)
        {
            // 转化为单通道灰度图，此时currentFrame已经存了tmpFrame的内容
            cvtColor(tmpFrame, currentFrame, CV_BGR2GRAY);
            currentFrame.convertTo(tmpFrameF, CV_32FC1);
            previousFrame.convertTo(previousFrameF, CV_32FC1);

            // 做差求绝对值
            absdiff(tmpFrameF, previousFrameF, currentFrameF);
            currentFrameF.convertTo(currentFrame, CV_8UC1);
            /*
            在currentFrameMat中找大于20（阈值）的像素点，把currentFrame中对应的点设为255
            此处阈值可以帮助把车辆的阴影消除掉
            */
//            threshold(currentFrameF, currentFrame, 20, 255.0, CV_THRESH_BINARY);
            threshold(currentFrame, currentFrame, 40, 255, CV_THRESH_BINARY);

            int g_nStructElementSize = 6; //结构元素(内核矩阵)的尺寸
            // 获取自定义核
            Mat element = getStructuringElement(MORPH_RECT,
                                                Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1),
                                                Point( g_nStructElementSize, g_nStructElementSize ));
            // 膨胀
            dilate(currentFrame, currentFrame, element);
            // 腐蚀
            //erode(currentFrame, currentFrame, element);
        }

        //把当前帧保存作为下一次处理的前一帧
        cvtColor(tmpFrame, previousFrame, CV_BGR2GRAY);

        // 显示图像
    
       imshow("Moving Area", currentFrame);
	
	cvtColor(tmpFrame , gray ,CV_BGR2GRAY);
		if (prvGray.data){
		  float flow_v;
		// pyr_scale：金字塔上下两层之间的尺度关系
		// levels：金字塔层数
		// winsize：均值窗口大小，越大越能denoise并且能够检测快速移动目标，但会引起模糊运动区域
		// iterations：迭代次数
		// poly_n：像素领域大小，一般为5，7等
		// poly_sigma：高斯标注差，一般为1-1.5
		// flags：计算方法。主要包括OPTFLOW_USE_INITIAL_FLOW和OPTFLOW_FARNEBACK_GAUSSIAN
			calcOpticalFlowFarneback(prvGray, gray, optFlow, 0.5, 3, 15, 3, 7, 1.2, 0);	//使用论文参数		
			flow_v=compute_absolute_mat2(optFlow,  absoluteFlow,      2        );
			normalize(absoluteFlow, img_for_show, 0, 255, NORM_MINMAX, CV_8UC1);
			threshold(img_for_show, img_for_show, 40, 255, THRESH_OTSU);
			imshow("opticalFlow", img_for_show);
			//spd check
			
			static float flow_dead=0.123,flow_v_sum;
			static int cnt_flow_sample;
		        static int flag_init;
		        #define force_stop_flow 1
			#define FLOW_SAM 20
			if(cnt_flow_sample++>FLOW_SAM&&flag_init==0){
			cnt_flow_sample=65535;flag_init=1;
			flow_dead=flow_v_sum/FLOW_SAM*2;
			}
			else
			flow_v_sum+=flow_v;   
			cout<<" flow_v:  "<<flow_v<<"    dead:  "<<flow_dead<<endl;
			int image_stable=0;
			if(flow_v<flow_dead||force_stop_flow)
			image_stable=1;
			if(image_stable){
			bitwise_and(img_for_show,currentFrame,and_image);
			int g_nStructElementSize =3	       ;
			Mat element = getStructuringElement(MORPH_RECT,Size(2 * g_nStructElementSize + 1, 2 * g_nStructElementSize + 1),Point( g_nStructElementSize, g_nStructElementSize ));
			morphologyEx(and_image, and_image, CV_MOP_CLOSE, element);
			}
			imshow("and_image",and_image);
			//blob detector
		      SimpleBlobDetector::Params params;
		      params.minDistBetweenBlobs = 0.0f;
		      params.filterByInertia = false;
		      params.filterByConvexity = false;
		      params.filterByColor = false;
		      params.filterByCircularity = false;
		      params.filterByArea = false;
		      // 声明根据面积过滤，设置最大与最小面积
		      params.filterByArea = true;
		      params.minArea = 20.0f;
		      params.maxArea = 800.0f;


		      SimpleBlobDetector detector(params);
		      std::vector<KeyPoint> keypoints;
		      detector.detect( and_image, keypoints);

		      Mat im_with_keypoints,show1;
		      tmpFrame.copyTo(im_with_keypoints);
		     // drawKeypoints( tmpFrame, keypoints, im_with_keypoints, Scalar(0,0,255),  DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
		      for (int i=0;i<keypoints.size();i++)
		      {
			Rect rect_draw;
			cv::Point pt;
			pt.x=keypoints[i].pt.x;
			pt.y=keypoints[i].pt.y;
			rect_draw.width =keypoints[i].size*2;
			//cout<<keypoints[i].size<<endl;
			rect_draw.height = rect_draw.width/1.23;
			rect_draw.x = pt.x-rect_draw.width/2;
			rect_draw.y = pt.y-rect_draw.height/2;
                       // cout<<pt<<endl;
			if(keypoints[i].size>10&&image_stable&&  
			  pt.y>0.05*rows&&pt.y<0.95*rows&&
			  pt.x>0.05*cols&&pt.x<0.95*cols
			)
			rectangle (im_with_keypoints,  rect_draw,Scalar(0, 255, 255),2, 1, 0);
		      }

		        vector<Feather> featherList;                    // 存放连通域特征
		       bwLabel(and_image, show1, featherList) ;
		       for (int i=0;i<featherList.size();i++)
		      {
			Rect rect_draw;
			cv::Point pt;
			pt.x=featherList[i].boundingbox.x+featherList[i].boundingbox.width/2;
			pt.y=featherList[i].boundingbox.y+featherList[i].boundingbox.height/2;
			int area_r=featherList[i].boundingbox.area();
			int check_r=rows*cols*0.001;
			//cout<<area_r<<"   "<<check_r<<endl;
			if(area_r>check_r&&featherList[i].area>222&&image_stable&&  
			  pt.y>0.05*rows&&pt.y<0.95*rows&&
			  pt.x>0.05*cols&&pt.x<0.95*cols)
			rectangle (im_with_keypoints,   featherList[i].boundingbox,  Scalar(0, 0, 255),2,1,0);
		      }
		      // Show blobs
		      imshow("keypoints", im_with_keypoints );
		}
		cv::swap(prvGray, gray);
	
           //imshow("Camera", tmpFrame);
        waitKey(1);
    }
}
