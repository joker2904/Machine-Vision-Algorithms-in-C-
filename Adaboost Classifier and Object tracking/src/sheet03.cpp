#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;
String face_cascade_name = "../face-model.xml";
string window_name = "Capture - Face detection";

bool FaceDetection(string imgPath)
{
  cv::Mat frame_gray = cv::imread(imgPath);   
  std::vector<Rect> faces;
  CascadeClassifier face_cascade;
  
  if( !face_cascade.load( face_cascade_name ) )
  { 
      cout<<"--(!)Error loading\n";
      return -1; 
  }
  
  //cvtColor( frame, frame_gray, CV_BGR2GRAY );
  //equalizeHist( frame_gray, frame_gray );

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse( frame_gray, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
  }
  //-- Show what you got
  imshow( window_name, frame_gray );
 }
 return true;
    
}



int main(int argc, char* argv[])
{
    // TODO implement your solution here
     
    // feel free to create additional .cpp files for other classes if needed
    FaceDetection("../img1.jpg");
    
    
    
    
    return 0;
}
