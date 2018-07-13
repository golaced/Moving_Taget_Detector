#include <iostream>
#include <fstream>
#include <sstream>   //istringstream ������������ͷ�ļ�
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include "kcftracker.hpp"
using namespace std;
using namespace cv;
int main()
{
	//�������
	//����Haar��LBP���������������  
	CascadeClassifier faceDetector;
	std::string faceCascadeFilename = "haarcascade_frontalface_default.xml";
	//�Ѻô�����Ϣ��ʾ  
	try{
		faceDetector.load(faceCascadeFilename);
	}
	catch (cv::Exception e){}
	if (faceDetector.empty())
	{
		std::cerr << "������������ܼ��� (";
		std::cerr << faceCascadeFilename << ")!" << std::endl;
		exit(1);
	}
	//int flags = CASCADE_SCALE_IMAGE;  //����������  
	Size minFeatureSize(30, 30);
	float searchScaleFactor = 1.1f;
	int minNeighbors = 4;
	std::vector<Rect> faces;

	//KCF
	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;
	int i = 0;
	Rect result;
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
 

	//��������ͷ  
	VideoCapture capture(0);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, 720);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 540);
	capture.set(CAP_PROP_FPS, 20);
	Mat frame, frame_gray;
	while (true)
	{		
		capture >> frame;
		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		if (i == 0)
		{
			int flags = CASCADE_FIND_BIGGEST_OBJECT | CASCADE_DO_ROUGH_SEARCH;    //ֻ�������������  
		//	double timeStart = (double)getTickCount();
			faceDetector.detectMultiScale(frame_gray, faces, searchScaleFactor, minNeighbors, flags, minFeatureSize);
		//	double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
			if (faces.size() > 0)
			{
				rectangle(frame, Point(faces[0].x, faces[0].y), Point(faces[0].x + faces[0].width, faces[0].y + faces[0].height), Scalar(0, 255, 255), 1, 8);//KCF ��ɫ
				i = 1;
				tracker.init(Rect(faces[0].x, faces[0].y, faces[0].width, faces[0].height), frame);
			}
			//cout << "�����������򹲺�ʱ��" << nTime << "��\n" << endl;
		}
		if (i>0)
		{
			result = tracker.update(frame);
			i++;
			rectangle(frame, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 255, 255), 1, 8);//KCF ��ɫ
			if (i > 20)
			{
				i = 0;
			}
		}
		cout << i << endl;

		imshow("ԭͼ", frame);
		waitKey(20);
	}
	waitKey();
	return 0;
}