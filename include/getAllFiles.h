#include <vector>//vector是一个能够存放任意类型的动态数组，能够增加和压缩数据。
using namespace std;
#ifndef _GETALLFILES_HPP_  //一个比较常用的C/C++杂注，只要在头文件的最开始加入这条杂注，就能够保证头文件只被编译一次。
#define _GETALLFILES_HPP_

void getAllFiles(string path, vector<string>& files);

#endif