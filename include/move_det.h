  #ifndef __MOVE_DET_H__
#define  __MOVE_DET_H__

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "OpencvInclude.h"


class Move_det
{
private:
	
	int Track_min_blob ;
	int out_window_dead;
	float blob_light_size;
	int en_tracker,en_image_show;
	int dead_coner;
	int number;
	char u_key[128];
	void compute_absolute_mat(const cv::Mat& in, cv::Mat & out,float thr);
	int isOverlap(const cv::Rect &rc1, const cv::Rect &rc2);
	float DecideOverlap(const cv::Rect &r1,const cv::Rect &r2);
	cv::Rect  rectCenterScale(cv::Rect rect, cv::Size size);

public:
      
	void init_detector(int tb=10, int od=88, float bs=0.7, int dc=25, char tracker=1,char en_show=1,char key[20]={0});
	void set_key(char *key);
	int update(cv::Mat& in);
	std::vector<cv::Rect> move_targets;
};

#endif