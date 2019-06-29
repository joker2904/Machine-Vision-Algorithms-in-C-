/*
 * nr1.cc
 *
 *  Created on: May 5, 2014
 *      Author: richard
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdlib.h>
#include <string>

using namespace std;
using namespace cv;

bool FaceDetection(string imgPath,string model)
{
  Mat frame = imread(imgPath);  
  std::vector<Rect> faces;
  CascadeClassifier face_cascade;
 
  if( !face_cascade.load( model ) )
  { 
      cout<<"--(!)Error loading\n";
      return false; 
  }
  
  //-- Detect faces
  face_cascade.detectMultiScale( frame, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  for( size_t i = 0; i < faces.size(); i++ )
    rectangle(frame, Point(faces[i].x, faces[i].y) , Point(faces[i].x+ faces[i].width , faces[i].y+ faces[i].height) , Scalar( 0,255, 0) , 2);
 
 
  //-- Show the detected faces
  imshow( "Capture - Face detection", frame );
  waitKey(0); 
  return true;
}


int main(int argc, const char* argv[]) {
	// implement your solution for task 1 here
	
        if (argc != 3) {
		std::cout << "usage: " << argv[0] << " <model> <image>" << std::endl;
		exit(1);
	}
        
        FaceDetection( std::string(argv[2]) , std::string(argv[1]) );
	return 0;
}
