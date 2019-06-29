#ifndef FOREST_H
#define FOREST_H

#include "tree_utils.h"
#include "Tree.h"

/// implements a classification forest
class Forest
{
   public:
    Forest(int , TreeParam& );
    void Train(std::vector<cv::Mat>* , std::vector<cv::Mat>*, int);
    double testImage(cv::Mat&, cv::Mat&);
    

  private:
    void initColors();
    std::vector<Tree> TreesInForest;
    TreeParam param;
    std::vector<cv::Scalar> colorClasses;
  //  void MultithreadFacilitate(void*);
};



/*
struct packet
{
  int i;
  int NoOfClasses;
  std::vector<cv::Mat>* trainImgs;
  std::vector<cv::Mat>* trainSegMaps;  
};

*/
#endif // FOREST_H


