#include "Forest.h"
#include <pthread.h>
#include <iostream>

using namespace std;


Forest::Forest(int NoOfTrees, TreeParam& params) : param(params) 
{
    (this->TreesInForest).reserve(NoOfTrees);
    for (int index = 0; index < NoOfTrees; ++index) 
    {
        Tree tempTree;
        tempTree.setParam(&param);
        (this->TreesInForest).push_back( tempTree );
    }
    initColors();
}

/*
void Forest::MultithreadFacilitate(void* ptr)
{
  struct packet* temp = (struct packet*)ptr;
  int index =  temp->i;
  int NumberOfClasses = temp->NoOfClasses;
  std::vector<cv::Mat>* train = temp->trainImgs;
  std::vector<cv::Mat>* segMaps = temp->trainSegMaps;
  (this->TreesInForest)[index].Train(train, segMaps, NumberOfClasses);
}


void Forest::Train(std::vector<cv::Mat>* trainImgs, std::vector<cv::Mat>* trainSegMaps, int NoOfClasses) 
{
    cout<< "Training the Forests :::" << (this->TreesInForest).size() << " trees." << std::endl;
    //std::vector<pthread_t> threads( (this->TreesInForest).size() );
    
    for (int i = 0; i < (this->TreesInForest).size(); ++i) 
    {
        struct packet* message ;
        message->i = i;
        message->NoOfClasses = NoOfClasses;
        message->trainImgs = trainImgs;
        message->trainSegMaps = trainSegMaps;
        
        pthread_create( &threads[i], NULL, Forest::MultithreadFacilitate , (void*) message);
 
    }
    
    // Join the threads after creation 
    for (int i = 0; i < (this->TreesInForest).size(); ++i) 
       pthread_join( threads[i], NULL);
}
*/

void Forest::Train(std::vector<cv::Mat>* trainImgs, std::vector<cv::Mat>* trainSegMaps, int NoOfClasses) 
{
    cout<< "Training the Forests :::" << (this->TreesInForest).size() << " trees." << std::endl;
   
    for (int i = 0; i < (this->TreesInForest).size(); ++i) 
    {
       (this->TreesInForest)[i].Train(trainImgs, trainSegMaps, NoOfClasses);
    }

}


double Forest::testImage(cv::Mat& testImg, cv::Mat& segmentMapOut)
{
    segmentMapOut.create(testImg.rows, testImg.cols, testImg.type());   
      
    for (int col = 0; col < (testImg.cols - param.ImagePatchDimensions); ++col)
    {
       for (int row = 0; row < (testImg.rows - param.ImagePatchDimensions); ++row)
       {
         std::vector<double> probabilities;
         probabilities.resize(param.numOfClasses);
         for (int i = 0; i < TreesInForest.size(); ++i) 
         {
            cv::Rect bbox(col, row, param.ImagePatchDimensions, param.ImagePatchDimensions);
            Sample s(&testImg, bbox, -1);
            TreesInForest[i].classifySample(s, probabilities);
         }
         int winnerClass = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));

         int x =  col + param.ImagePatchDimensions / 2.0f;
         int y =  row + param.ImagePatchDimensions / 2.0f;

         segmentMapOut.at<cv::Vec3b>(y, x)[0] = colorClasses[winnerClass][0];
         segmentMapOut.at<cv::Vec3b>(y, x)[1] = colorClasses[winnerClass][1];
         segmentMapOut.at<cv::Vec3b>(y, x)[2] = colorClasses[winnerClass][2];

       }
    }
    
}



void Forest::initColors() 
{
    colorClasses.resize(param.numOfClasses);
    if(param.numOfClasses == 4) 
    {
        colorClasses[0] = 0; //black
        colorClasses[1] = cv::Scalar(0, 0, 255); // red
        colorClasses[2] = cv::Scalar(255, 0, 0); // blue
        colorClasses[3] = cv::Scalar(0, 255, 0); // green

    }
    else
    {
        int64 state = time(NULL);
        cv::RNG rng(state);
        for (int i = 0; i < colorClasses.size(); ++i)
        {
            cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            colorClasses[i] = color;
        }
    }
}