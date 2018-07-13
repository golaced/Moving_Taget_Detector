#include <io.h> //����_finddata_t
#include "getAllFiles.h"
void getAllFiles(string path, vector<string>& files)
{
	//�ļ�����  
	intptr_t  hFile = 0;
	//�ļ���Ϣ  
	struct _finddata_t fileinfo;  //�����õ��ļ���Ϣ��ȡ�ṹ
	string p;  //string��������˼��һ����ֵ����:assign()���кܶ����ذ汾
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))  //�ж��Ƿ�Ϊ�ļ���
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					files.push_back(p.assign(path).append("/").append(fileinfo.name));//�����ļ�������
					getAllFiles(p.assign(path).append("/").append(fileinfo.name), files);//�ݹ鵱ǰ�ļ���
				}
			}
			else    //�ļ�����
			{
				files.push_back(p.assign(path).append("/").append(fileinfo.name));//�ļ���
			}
		} while (_findnext(hFile, &fileinfo) == 0);  //Ѱ����һ��ɹ�����0������-1
		_findclose(hFile);
	}

	//{
	//	char * DistAll = "AllFiles.txt";
	//	ofstream ofn(DistAll);  //�����ļ���
	//	int  number = 0;
	//	ofn << number << endl;
	//	for (int i = 0; i < number; i++)
	//	{
	//		ofn << files[i] << endl;
	//	}
	//	ofn.close();
	//}


}