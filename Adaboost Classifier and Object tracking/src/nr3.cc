/*
 * nr3.cc
 *
 *  Created on: Apr 28, 2014
 *      Author: richard
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <sstream>
#include "Types.hh"
#include "AdaBoost.hh"
#include <stdio.h>
#include <malloc.h>

#define _objectWindow_width 121
#define _objectWindow_height 61

#define _searchWindow_width 61
#define _searchWindow_height 61

// use 30/15 for overlapping negative examples and 120/60 for non-overlapping negative examples
#define _displacement_x 120
#define _displacement_y 60

void computeHistogram(const cv::Mat& image, const cv::Point& p, Vector& histogram) {
  int histSize = 256;
  float range[] = { 0, 256 } ;
  const float* histRange = { range };
  bool uniform = true; bool accumulate = false;
  cv::Mat hist;
  cv::Mat temp;
  /// Compute the histograms:
  cv::calcHist( &image, 1, 0, temp, hist, 1, &histSize, &histRange, uniform, accumulate );
  //std::cout<<"\n"<<"   "<<hist.rows<<" "<<hist.cols;
  //cv::imshow("Segmented grayscaled image::", hist); 
  //cv::waitKey(0);
  for(u32 i = 0;i<256;++i)
      histogram.push_back( hist.at<float>(i,0) ); 
  
}

void generateTrainingData(std::vector<Example>& data, const std::vector<cv::Mat>& imageSequence, const std::vector<cv::Point>& referencePoints) {
     for( u32 i = 0; i < imageSequence.size(); ++i)
     {
         Example temp;
         computeHistogram(imageSequence.at(i),referencePoints.at(i),temp.attributes);
         temp.label = i%5==0 ? 1:0;
         data.push_back(temp);
     }
    
}

void loadImage(const std::string& imageFile, cv::Mat& image,u32 column,u32 row) {
                image = cv::imread(imageFile);
                //std::cout<<"\n"<<row<<" "<<column<<" "<<image.rows<<" "<<image.cols;
                cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
                //cv::rectangle(image, cv::Point(10, 10) , cv::Point(300 , 100) , cv::Scalar( 0,255, 0) , 2);
                image = image( cv::Rect(column,row, _objectWindow_width,_objectWindow_height) & cv::Rect(0, 0, image.cols, image.rows) ); // using a rectangle
                //cv::imshow("Segmented grayscaled image::", image); 
                //cv::waitKey(0); 
}

void loadTrainFrames(const char* trainDataFile, std::vector<cv::Mat>& imageSequence,std::vector<cv::Point>& referencePoints) {
     std::ifstream f(trainDataFile);
     std::string file;
     u32 column,row;
     cv::Mat image;
     int distancex = _displacement_x;
     int distancey = _displacement_y;
     for (u32 i = 0; i < 10; i++) {
		f >> file;
		f >> column;
                f >> row;
                // Non-overlapping training set of classifiers
                int x = column-(_objectWindow_width/2);
                int y = row - (_objectWindow_height);
                // loading frames for positive examples
                loadImage("../nemo/"+file,image,x,y);
                imageSequence.push_back(image);
                referencePoints.push_back( cv::Point(column,row) );   
                
                //loading frames for negative examples ( 8 negative examples per positive example ), around the positive example
             
		//loadImage("../nemo/"+file,image,x-distancex,y-distancey);
                //imageSequence.push_back(image);
                //referencePoints.push_back( cv::Point(x-distancex,y-distancey) );   
               
                //loadImage("../nemo/"+file,image,x-distancex,y+distancey);
                //imageSequence.push_back(image);
                //referencePoints.push_back( cv::Point(x-distancex,y+distancey) );
              
                loadImage("../nemo/"+file,image,x-distancex,y);
                imageSequence.push_back(image);
                referencePoints.push_back( cv::Point(x-distancex,y) );  
          
                
                
                //loadImage("../nemo/"+file,image,x+distancex,y-distancey);
                //imageSequence.push_back(image);
                //referencePoints.push_back( cv::Point(x+distancex,y-distancey) );  
                
                loadImage("../nemo/"+file,image,x+distancex,y);
                imageSequence.push_back(image);
                referencePoints.push_back( cv::Point(x+distancex,y) );   
               
                //loadImage("../nemo/"+file,image,x+distancex,y+distancey);
                //imageSequence.push_back(image);
                //referencePoints.push_back( cv::Point(x+distancex,y+distancey) );
              
                
                
                loadImage("../nemo/"+file,image,x,y-distancey);
                imageSequence.push_back(image);
                referencePoints.push_back( cv::Point(x,y-distancey) );  
          
                loadImage("../nemo/"+file,image,x,y+distancey);
                imageSequence.push_back(image);
                referencePoints.push_back( cv::Point(x,y+distancey) );  
                
                
	         
	}
}
	
 void getNewTrainingDataset(cv::Mat image,std::vector<cv::Mat>& NewTrainingDataSet,cv::Point lastPosition,std::vector<cv::Point>& referencePoints)
 {
                int x = lastPosition.x - (_objectWindow_width/2);
                int y = lastPosition.y - (_objectWindow_height);
                int distancex = _displacement_x;
                int distancey = _displacement_y;
                // loading frames for positive examples
                
                NewTrainingDataSet.push_back(image( cv::Rect(x,y, _objectWindow_width,_objectWindow_height)) );
                referencePoints.push_back( cv::Point(x,y) );   
                
                //loading frames for negative examples ( 8 negative examples per positive example ), around the positive example                	              
                NewTrainingDataSet.push_back(image( cv::Rect(x-distancex,y, _objectWindow_width,_objectWindow_height)) );
                referencePoints.push_back( cv::Point(x-distancex,y) );  
          
                NewTrainingDataSet.push_back(image( cv::Rect(x+distancex,y, _objectWindow_width,_objectWindow_height)) );
                referencePoints.push_back( cv::Point(x+distancex,y) );   
               
                NewTrainingDataSet.push_back(image( cv::Rect(x,y-distancey, _objectWindow_width,_objectWindow_height)) );
                referencePoints.push_back( cv::Point(x,y-distancey) );  
          
                NewTrainingDataSet.push_back(image( cv::Rect(x,y+distancey, _objectWindow_width,_objectWindow_height)) );
                referencePoints.push_back( cv::Point(x,y+distancey) );  
                
 }
     
     
void loadTestFrames(const char* testDataFile, std::vector<cv::Mat>& imageSequence, cv::Point& startingPoint) {
     std::ifstream f( testDataFile);
     std::string file;
     cv::Mat image;
     f>>startingPoint.x;
     f>>startingPoint.y;
     for (u32 i = 0; i < 22; i++) {
		f >> file;
		// loading frames for positive examples,
                image = cv::imread("../nemo/"+file);
                cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
                imageSequence.push_back(image);
     }
}


void findBestMatch(std::vector<Example>& data,const cv::Mat& image, cv::Point& lastPosition, AdaBoost& adaBoost) {
    int startx = lastPosition.x - 31;
    int starty = lastPosition.y - 31;
  
    f32 mc = -1.0f;
    cv::Point bestPoint;
    
        
    //std::cout<<std::endl<<std::endl;
    for( int i = startx ; i < (startx + 61); ++i) // < image.rows ?  startx+61: image.rows); ++i)
    {
       //std::cout<<std::endl;
       for( int j = starty ; j < (starty + 61); ++j) // < image.rows ?  startx+61: image.rows); ++i)
       {
         Example temp;
         // Vector testhistogram; 
         cv::Mat imagePatch = image( cv::Rect(i,j, _objectWindow_width,_objectWindow_height)   & cv::Rect(0, 0, image.cols, image.rows) );
         
         computeHistogram(imagePatch,cv::Point(i,j),temp.attributes);
         u32 testlabel = adaBoost.classify(temp.attributes);
         //std::cout<<testlabel;
         if( testlabel == 1)
         {
         f32 conf = adaBoost.confidence(temp.attributes, 1);
         temp.label = 1;
          if(conf > mc)
          {
             bestPoint.x = i+25;
             bestPoint.y = j+31;
             mc = conf;
          }
         }
         else
             temp.label = 0;
        data.push_back(temp);
        }
        if(mc > 0.5)
          lastPosition = bestPoint;
       
    }
 
}

void drawTrackedFrame(cv::Mat& image, cv::Point& position) {
	cv::rectangle(image, cv::Point(position.x - _objectWindow_width / 2, position.y - _objectWindow_height / 2),
			cv::Point(position.x + _objectWindow_width / 2, position.y + _objectWindow_height / 2), 0, 3);
        
        //cv::rectangle(image, cv::Point(position.x , position.y ),
	//		cv::Point(position.x + _objectWindow_width , position.y + _objectWindow_height ), 0, 3);
	cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE);
	cv::imshow("Display window", image);
	//std::sleep(1);
	cv::waitKey(0);
}

int main( int argc, char** argv )
{

	//implement the functions above to make this work. Or implement your solution from scratch
	if(argc != 5) {
		std::cout <<" Usage: " << argv[0] << " <training-frame-file> <test-frame-file> <# iterations for AdaBoost> <#k>" << std::endl;
		return -1;
	}

	u32 adaBoostIterations = atoi(argv[3]);
        u32 k = atoi(argv[4]);
	// load the training frames
	std::vector<cv::Mat> imageSequence;
	std::vector<cv::Point> referencePoints;
	loadTrainFrames(argv[1], imageSequence, referencePoints);
      	// generate gray-scale histograms from the training frames:
	// one positive example per frame (_objectWindow_width x _objectWindow_height window around reference point for object)
	// four negative examples per frame (with _displacement_{x/y} + small random displacement from reference point)
	std::vector<Example> trainingData;
	generateTrainingData(trainingData, imageSequence, referencePoints);
        // initialize AdaBoost and train a cascade with the extracted training data
	AdaBoost adaBoost(adaBoostIterations,k);
	adaBoost.initialize(trainingData);
	adaBoost.trainCascade(trainingData);

	// log error rate on training set
	u32 nClassificationErrors = 0;
        std::cout << "Classification Sequence on Traing Set :: "<<std::endl;
	for (u32 i = 0; i < trainingData.size(); i++) {
		u32 label = adaBoost.classify(trainingData.at(i).attributes);
		nClassificationErrors += (label == trainingData.at(i).label ? 0 : 1);
                std::cout<<"label="<<label<<" :::"<<(label == trainingData.at(i).label ? "Pass\n" : "Fail\n");
	}
	std::cout << "Error rate on training set: " << (f32)nClassificationErrors / (f32)trainingData.size() << std::endl;

	// load the test frames and the starting position for tracking
	std::vector<Example> testImages;
        std::vector<cv::Mat> TestimageSequence;
        //std::vector<cv::Mat> NewTrainingImages;
	cv::Point lastPosition;
        //std::vector<cv::Point> ReferencePoints;
        std::vector<Example> newtrainingData;
	loadTestFrames(argv[2], TestimageSequence, lastPosition); // contains the full image-sequence list and the starting point for the first image 
       	// for each frame...
        drawTrackedFrame(TestimageSequence.at(0), lastPosition);
         
	for (u32 i = 1; i < TestimageSequence.size(); i++) {
		// ... find the best match in a window of size
		// _searchWindow_width x _searchWindow_height around the last tracked position
             
                findBestMatch(newtrainingData,TestimageSequence.at(i), lastPosition, adaBoost);
                drawTrackedFrame(TestimageSequence.at(i), lastPosition);
                
                // The following steps implement ensemble tracking
                //getNewTrainingDataset(TestimageSequence.at(i),NewTrainingImages,lastPosition,ReferencePoints);
                //generateTrainingData(newtrainingData,NewTrainingImages,ReferencePoints);
                adaBoost.retrain(newtrainingData);
		// draw the result
		
                std::cout<<"\nSTarting Position for tracking :::: "<<lastPosition.x<<"  "<<lastPosition.y;
	}

	return 0;
}
