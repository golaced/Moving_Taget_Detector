
#include "move_det.h"
#include "ukey.h"
#include "back_ground.h"
#include <select.h>
#include "kcftracker.hpp"
using namespace std;
using namespace cv;

void Move_det:: init_detector(int tb, int od, float bs, int dc, char tracker,char en_show,char key[20])
{
   //cout<<"Lis: "Lisence<<endl;
   for(int i=0;i<20;i++)
      u_key[i]=key[i];
 
   Track_min_blob=tb;
    out_window_dead=od;
    blob_light_size=bs;
    dead_coner=dc;
    en_tracker=tracker;	
    en_image_show=en_show;
    number=0;
}
#define MAX_PRO_FRAME 3000
int  Move_det:: update(cv::Mat& in)
{
    static int pro_frame,key_right=0;
    int i,w=640,h=480;
    int dead_coner=25;
    Mat gray,prvGray, optFlow ,absoluteFlow, img_for_show,rectImage,show1;
    Rect rect;
    vector<Feather> featherList;                    // 存放连通域特征
    vector<Rect> detect_target;
    vector<Rect> detect_target_final;
    vector<Point2d> coner_reg;
    Mat element = getStructuringElement(MORPH_RECT, Size(  2,2 ));
    Mat element2 = getStructuringElement(MORPH_RECT, Size(12,12 ));
    static  vector<Point> trajectory[10];
    static  Rect result[10];
    static  int kcf_flag[10];
    static  int unmove_kcf_flag[10];
       //KCF
    bool HOG = true;
    bool FIXEDWINDOW = true;
    bool MULTISCALE = false;
    bool SILENT = true;
    bool LAB = false;
    
    static OriginalVibe vibe(66,   2,   35,  5,  8,   8);
    static  KCFTracker tracker1(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    static KCFTracker tracker2(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    static KCFTracker tracker3(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    static KCFTracker tracker4(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    static  KCFTracker tracker5(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    static KCFTracker tracker6(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    static  KCFTracker tracker7(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    static KCFTracker tracker8(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    static KCFTracker tracker9(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    static KCFTracker tracker10(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    static int init=0;
    if(!init){init=1;
    key_right= set_lisence(u_key);
    }
    //-------------------------------------------------------------------
    Mat capIn,frame,seg;
    pro_frame++;
    if( pro_frame> MAX_PRO_FRAME||key_right==0)
    {
      pro_frame=MAX_PRO_FRAME+2;
      cout<<"Demo End!  Need U-key from Littro "<<endl;
      goto end1;
    }
	        move_targets.clear();
		
		in.copyTo(frame); 	
		if(! frame.data)goto end1;
		w=frame.cols;
		h=frame.rows;
		frame.copyTo(capIn);
	        cv::cvtColor(frame, frame, CV_BGR2GRAY);
		number++;
		if(number == 1)
		{     	vibe.originalVibe_Init_GRAY( frame );
		        cout<<"Modle Re-Init!!!"<<endl;
			goto end1;
		}
		else if(number % 400==0)
		{  
		      //vibe.originalVibe_Init_GRAY( frame );	
		     cout<<"Modle Re-Init!!!"<<endl;
		}
	      //background modle
	      vibe.originalVibe_ClassifyAndUpdate_GRAY(frame,seg);
	      erode(seg, seg, element);
	      dilate(seg, seg, element2);
		///-------------------  Find Target ----------------				

		       bwLabel(seg, show1, featherList) ;
		       for (int i=0;i<featherList.size();i++)
		      {
			Rect rect_draw;
			cv::Point pt;
			pt.x=featherList[i].boundingbox.x+featherList[i].boundingbox.width/2;
			pt.y=featherList[i].boundingbox.y+featherList[i].boundingbox.height/2;
			int area_r=featherList[i].boundingbox.area();
			if(featherList[i].area>60){
			   int has_white=check_bw(capIn,featherList[i].boundingbox,Scalar(255,255,255),blob_light_size,200);
			   if(has_white){
			      detect_target.push_back(featherList[i].boundingbox);
			  // rectangle (capIn,   featherList[i].boundingbox,  Scalar(0, 0, 255),1,1,0);
			   }
			 }
		      }	
		      //--------------------clip overlap tangle

		       for (int i=0;i<detect_target.size();i++) {
			   vector<Rect> detect_target_templ;
			   int good1=1;
			   for(int j=0;j<coner_reg.size();j++)
			   {
			     int coner_erox=abs(detect_target[i].x-detect_target[i].width/2-coner_reg[j].x);
			     int coner_eroy=abs(detect_target[i].y-detect_target[i].height/2-coner_reg[j].y);
			     if(coner_erox<dead_coner&&coner_eroy<dead_coner)
			        good1=0;
			   }
			   if(good1){
			    for (int j=0;j<detect_target.size();j++)
			      {			     
				  int coner_erox=(detect_target[i].x-detect_target[i].width/2)-(detect_target[j].x-detect_target[j].width/2);
				  int coner_eroy=(detect_target[i].y-detect_target[i].height/2)-(detect_target[j].y-detect_target[j].height/2);
				  if(abs(coner_erox)<dead_coner&&abs(coner_eroy)<dead_coner)
				    detect_target_templ.push_back(detect_target[j]);
			      }
			      int max_size=0;
			      int max_id=65535; 
			      for (int j=0;j<detect_target_templ.size();j++)
			      {			     
				  if(detect_target_templ[j].width*detect_target_templ[j].height>max_size)
				  {
				    max_size=detect_target_templ[j].width*detect_target_templ[j].height;
				    max_id=j; 
				  }
			      } 
			      if(max_id!=65535){
				  coner_reg.push_back(Point2d( detect_target_templ[max_id].x-detect_target_templ[max_id].width/2,
										      detect_target_templ[max_id].y-detect_target_templ[max_id].height/2   ));
				  detect_target_final.push_back(detect_target_templ[max_id]);
			      }
			   }
		       } 
		       
		      for (int i=0;i<detect_target_final.size();i++)
			if(!en_tracker){
		        rectangle (capIn,   detect_target_final[i],  Scalar(0, 255, 255),2,1,0);
		        move_targets.push_back(detect_target_final[i]);
			}
    
		      //tracker
		      static int track_init=0;
		      static int track_reinit_cnt[10];
		      static int track_can_init_flag[10];
		      if(detect_target_final.size()){
		      for(int j=0;j<detect_target_final.size();j++){  
				  //check track cover rect
				  int is_cover=0;
				  for(int k=0;k<10;k++)
				  {
				      if(kcf_flag[k]==1){
					float over_rate=DecideOverlap(detect_target_final[j],result[k]);
					if(over_rate>0.001)
					   is_cover=1;
					//cout<<"is_cover:  "<<is_cover<<endl;
				      }
				  }
				if(is_cover==0){
						for(int i=0;i<10;i++){
						          int rx=detect_target_final[j].x+detect_target_final[j].width/2;
							  int ry=detect_target_final[j].y+detect_target_final[j].height/2;
							  int size_rect=detect_target_final[j].width*detect_target_final[j].height;
							  int track_mask=0;
							  if(rx>0.2*w&&rx<0.8*w&&ry>0.2*h&&ry<0.8*h&&size_rect>Track_min_blob*Track_min_blob)
							      track_mask=1;
							  if(size_rect<Track_min_blob*Track_min_blob*2)
							      track_mask=1;
							  if(kcf_flag[i]==0&&track_can_init_flag[i]==0&&track_mask
							  ){
								    if(size_rect>Track_min_blob*Track_min_blob){
								    rect=detect_target_final[j];
								    //cout<<i<<" track init :"<<rect<<endl;
								    rectImage=capIn(rect); //子图像显示  
								    Size size_up;
								    if(rect.width<65&&rect.height<45){
								    size_up.width=rect.width*0.5;
								    size_up.height=rect.height*0.5;
								    rect=rectCenterScale(rect,size_up);
								    }
								   // imshow("Sub Image",rectImage); 		
								    kcf_flag[i]=1;	
									      //kcftracker init
									      switch(i){
										case  0: tracker1.init(rect, capIn);break;
										case  1: tracker2.init(rect, capIn);break;
										case  2: tracker3.init(rect, capIn);break;
										case  3: tracker4.init(rect, capIn);break;
										case  4: tracker5.init(rect, capIn);break;
										case  5: tracker6.init(rect, capIn);break;
										case  6: tracker7.init(rect, capIn);break;
										case  7: tracker8.init(rect, capIn);break;
										case  8: tracker9.init(rect, capIn);break;
										case  9: tracker10.init(rect, capIn);break;
									      }
								      goto jump1;
								      }
							  }
						}	
				}
		      }
		      }
		      jump1:;
  
		      //update track
		      for(int i=0;i<10;i++){
			  if(kcf_flag[i] &&en_tracker) {
				    switch(i){
						  case  0: result[0] =tracker1.update(capIn);break;
						  case  1: result[1] =tracker2.update(capIn);break;
						  case  2: result[2] =tracker3.update(capIn);break;
						  case  3: result[3] =tracker4.update(capIn);break;
						  case  4: result[4] =tracker5.update(capIn);break;
						  case  5: result[5] =tracker6.update(capIn);break;
						  case  6: result[6] =tracker7.update(capIn);break;
						  case  7: result[7] =tracker8.update(capIn);break;
						  case  8: result[8] =tracker9.update(capIn);break;
						  case  9: result[9] =tracker10.update(capIn);break;
						}
				    rectangle(capIn, Point(result[i].x, result[i].y) , 
					      Point(result[i].x + result[i].width, result[i].y + result[i].height), Scalar(55+15*i, 155-25*i, 255), 2);//KCF »ÆÉ«
			   trajectory[i].push_back(Point(result[i].x+result[i].width/2,result[i].y+result[i].height/2));
			   Rect temp(Point(result[i].x, result[i].y) ,   Point(result[i].x + result[i].width, result[i].y + result[i].height));
			   move_targets.push_back(temp);
			    for(int k=0;k<trajectory[i].size()-1;k++)
			   {
			  	  line(capIn,trajectory[i][k],trajectory[i][k+1],Scalar(55+15*i, 155-25*i, 255), 2);
			   }
			    }
			}
				
		        //------------------------release track
		        static int cnt_out_window[10];
		         for(int i=0;i<10;i++)
			 {
				      if(kcf_flag[i]==1){
					int cx=result[i].x + result[i].width/2;
					int cy=result[i].y + result[i].height/2;
					if((cx<out_window_dead||cx>w-out_window_dead||cy<out_window_dead||cy>h-out_window_dead)
					  &&cx!=0&&cy!=0)
					  cnt_out_window[i]++;
					if(cnt_out_window[i]>5)
					{ cnt_out_window[i]=unmove_kcf_flag[i]=kcf_flag[i]=0;
					  result[i].x=result[i].y=result[i].width=result[i].height=0;
					  trajectory[i].clear();track_can_init_flag[i]=1;track_reinit_cnt[i]=0;
					 // cout<<i<<" track kill: "<<cx<<" "<<cy<<endl;
					  
					}
				      }
			 } 
			 //kick over_lap track
			  for(int i=0;i<10;i++)
			  {  
			     for(int k=0;k<10;k++){
			      int is_cover=0;
			      if(kcf_flag[k]==1&&kcf_flag[i]==1&&i!=k){
				Size size_up;
				size_up.width=result[k].width*1;
				size_up.height=result[k].height*1;
				float over_rate=DecideOverlap(result[i],result[k]);
				if(over_rate>0.001){
					 cnt_out_window[i]= unmove_kcf_flag[i]=kcf_flag[i]=0;
					  result[i].x=result[i].y=result[i].width=result[i].height=0;
					  trajectory[i].clear();track_can_init_flag[i]=1;track_reinit_cnt[i]=0;
					  //cout<<k<<" track kill: "<<endl;
				  
				}  
			       }
			     }
			  }
			  //kick unmove track
			   static int pos[2][10];
			   static int cnt1;
			  if(cnt1++>10){cnt1=0;
			  for(int i=0;i<10;i++)
			  {  
			    if(kcf_flag[i]==1)
			    {
			      int move_dis=abs(pos[0][i]-result[i].x)+abs(pos[1][i]-result[i].y);
			      //cout<<"move_dis:  "<<move_dis<<endl;
			      if(move_dis<=3&&result[i].width*result[i].height<40*40)
				    unmove_kcf_flag[i]++;
			      else   if(move_dis<=25&&result[i].width*result[i].height>40*40)
				    unmove_kcf_flag[i]++;
			      else
				    unmove_kcf_flag[i]=0;
			       pos[0][i]=result[i].x;
			       pos[1][i]=result[i].y;
			         if(unmove_kcf_flag[i]>2)
				    {
					  cnt_out_window[i]=unmove_kcf_flag[i]=kcf_flag[i]=0;
					  result[i].x=result[i].y=result[i].width=result[i].height=0;
					  trajectory[i].clear();track_can_init_flag[i]=1;track_reinit_cnt[i]=0;
					  //cout<<i<<" track kill: "<<endl;
				    }
			    }
			  }}
		       //draw detection witout overlap	  
		       static int cnt2;
		       if(cnt2++>3){cnt2=0;
			for (int i=0;i<detect_target_final.size();i++){
			  int temp=0;
			  for(int k=0;k<10;k++){
			   float over_rate=DecideOverlap(detect_target_final[i],result[k]);
			   if(over_rate>0.001)
			   {temp=1;break;}
			  }
			  if(temp==0)
		             rectangle (capIn,   detect_target_final[i],  Scalar(0, 255, 255),1,1,0);	  
			}
		       }
		       for(i=0;i<10;i++){
			 if(track_can_init_flag[i])
			      track_reinit_cnt[i]++;
		         if( track_reinit_cnt[i]>30)
			 { track_reinit_cnt[i]=0;track_can_init_flag[i]=0;}
		       }
		     
   if(en_image_show)
      imshow("orignal",capIn);
    end1:;
   return move_targets.size();
}

//----------------------------------------------
void Move_det:: compute_absolute_mat(const Mat& in, Mat & out,float thr)
{
	if (out.empty()){
		out.create(in.size(), CV_32FC1);
	}
 
	const Mat_<Vec2f> _in = in;
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
			
		}
	}
}

int Move_det::isOverlap(const Rect &rc1, const Rect &rc2)
{
    if (rc1.x + rc1.width  > rc2.x &&
        rc2.x + rc2.width  > rc1.x &&
        rc1.y + rc1.height > rc2.y &&
        rc2.y + rc2.height > rc1.y
       )
        return 1;
    else
        return 0;
}


float Move_det::DecideOverlap(const Rect &r1,const Rect &r2)
{
	int x1 = r1.x;
	int y1 = r1.y;
	int width1 = r1.width;
	int height1 = r1.height;
 
	int x2 = r2.x;
	int y2 = r2.y;
	int width2 = r2.width;
	int height2 = r2.height;
 
	int endx = max(x1+width1,x2+width2);
	int startx = min(x1,x2);
	int width = width1+width2-(endx-startx);
 
	int endy = max(y1+height1,y2+height2);
	int starty = min(y1,y2);
	int height = height1+height2-(endy-starty);
 
	float ratio = 0.0f;
	float Area,Area1,Area2;
 
	if (width<=0||height<=0)
	    return 0.0f;
	else
	{
		Area = width*height;
		Area1 = width1*height1;
		Area2 = width2*height2;
		ratio = Area /(Area1+Area2-Area);
	}
 
	return ratio;
}
	
Rect Move_det:: rectCenterScale(Rect rect, Size size)
{
	rect = rect + size;	
	Point pt;
	pt.x = cvRound(size.width/2.0);
	pt.y = cvRound(size.height/2.0);
	return (rect-pt);
}



