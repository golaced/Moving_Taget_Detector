#include "select.h"
#include <stack>
#include <vector>
using namespace cv;
using namespace std;


int check_bw(cv::Mat& image, cv::Rect rectin,cv::Scalar color,float thr,int dis)
{
  int cnt=0,i,j;
  int bx=rectin.x-rectin.width/2;
  int by=rectin.y-rectin.height/2;
  int mx=rectin.x+rectin.width/2;
  int my=rectin.y+rectin.height/2;

  
     int size=abs(my-by)*abs(bx-mx);
     //cout<<"si:"<<size<<endl;
  for (i=bx;i<mx;i++)
      for (j=by;j<my;j++){
		int cPointB=image.at<Vec3b>(j,i)[0];  
	        int cPointG=image.at<Vec3b>(j,i)[1];  
	        int cPointR=image.at<Vec3b>(j,i)[2];  
		int dist=abs(cPointB - color[0])/3 + abs(cPointG - color[1])/3 + abs(cPointR - color[2])/3;
		if(dist<dis)
		   cnt++;
      }
 //  cout<<"cnt:"<<cnt<<endl;   
   if(cnt>thr*size)
     return 1;
   else 
     return 0;
}
/* 
Input: 
    src: 待检测连通域的二值化图像
Output:
    dst: 标记后的图像
    featherList: 连通域特征的清单
return： 
    连通域数量。
*/
int bwLabel(Mat & src, Mat & dst, vector<Feather> & featherList)
{
    int rows = src.rows;
    int cols = src.cols;

    int labelValue = 0;
    Point seed, neighbor;
    stack<Point> pointStack;    // 堆栈

    int area = 0;               // 用于计算连通域的面积
    int leftBoundary = 0;       // 连通域的左边界，即外接最小矩形的左边框，横坐标值，依此类推
    int rightBoundary = 0;
    int topBoundary = 0;
    int bottomBoundary = 0;
    Rect box;                   // 外接矩形框
    Feather feather;

    featherList.clear();    // 清除数组

    dst.release();
    dst = src.clone();
    for( int i = 0; i < rows; i++)
    {
        uchar *pRow = dst.ptr<uchar>(i);
        for( int j = 0; j < cols; j++)
        {
            if(pRow[j] == 255)
            {
                area = 0;
                labelValue++;           // labelValue最大为254，最小为1.
                seed = Point(j, i);     // Point（横坐标，纵坐标）
                dst.at<uchar>(seed) = labelValue;
                pointStack.push(seed);

                area++;
                leftBoundary = seed.x;
                rightBoundary = seed.x;
                topBoundary = seed.y;
                bottomBoundary = seed.y;

                while(!pointStack.empty())
                {
                    neighbor = Point(seed.x+1, seed.y);
                    if((seed.x != (cols-1)) && (dst.at<uchar>(neighbor) == 255))
                    {
                        dst.at<uchar>(neighbor) = labelValue;
                        pointStack.push(neighbor);

                        area++;
                        if(rightBoundary < neighbor.x)
                            rightBoundary = neighbor.x;
                    }

                    neighbor = Point(seed.x, seed.y+1);
                    if((seed.y != (rows-1)) && (dst.at<uchar>(neighbor) == 255))
                    {
                        dst.at<uchar>(neighbor) = labelValue;
                        pointStack.push(neighbor);

                        area++;
                        if(bottomBoundary < neighbor.y)
                            bottomBoundary = neighbor.y;

                    }

                    neighbor = Point(seed.x-1, seed.y);
                    if((seed.x != 0) && (dst.at<uchar>(neighbor) == 255))
                    {
                        dst.at<uchar>(neighbor) = labelValue;
                        pointStack.push(neighbor);

                        area++;
                        if(leftBoundary > neighbor.x)
                            leftBoundary = neighbor.x;
                    }

                    neighbor = Point(seed.x, seed.y-1);
                    if((seed.y != 0) && (dst.at<uchar>(neighbor) == 255))
                    {
                        dst.at<uchar>(neighbor) = labelValue;
                        pointStack.push(neighbor);

                        area++;
                        if(topBoundary > neighbor.y)
                            topBoundary = neighbor.y;
                    }

                    seed = pointStack.top();
                    pointStack.pop();
                }
                box = Rect(leftBoundary, topBoundary, rightBoundary-leftBoundary, bottomBoundary-topBoundary);
                rectangle(dst, box, 255);
                feather.area = area;
                feather.boundingbox = box;
                feather.label = labelValue;
                featherList.push_back(feather);
            }
        }
    }
    return labelValue;
}