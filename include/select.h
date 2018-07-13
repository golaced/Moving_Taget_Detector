#ifndef SELECT_H_
#define SELECT_H_

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cv.h"
#include <cstdio>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cxcore.h>

typedef struct _Feather
{
    int label;              // 连通域的label值
    int area;               // 连通域的面积
    cv::Rect boundingbox;       // 连通域的外接矩形框
} Feather;

int bwLabel(cv::Mat & src, cv::Mat & dst,cv:: vector<Feather> & featherList);
int check_bw(cv::Mat& image, cv::Rect rectin,cv::Scalar color,float thr,int dis);

#endif /* SELECT_H_ */
