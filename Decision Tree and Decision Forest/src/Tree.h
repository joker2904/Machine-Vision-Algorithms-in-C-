#ifndef TREE_H
#define TREE_H
#include <tree_utils.h>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "Node.h"
#include "BinaryTest.h"
/// Implements the classification tree
class Tree
{
 
  private:
    TreeParam param;
    Node* root ;
    std::vector<BinaryTest> getBinaryTests();
    std::vector<cv::Scalar> colorClasses;

  public: 
  Tree()
  {
      isTreeTrained = false;
      root = NULL;
  }
  ~Tree();
  
  
  
  
  bool isTreeTrained;    
  void setParam(TreeParam*);
  void Train(std::vector<cv::Mat>*,std::vector<cv::Mat>* ,int);
  std::vector<BinaryTest> GenerateRandomBinaryTests(void);
  bool isTrained(void);
  void testImage(cv::Mat&,cv::Mat&);
  void ConstructTrainingSamples(std::vector<cv::Mat>* ,
                                std::vector<cv::Mat>* ,
                                std::vector<Sample>*  , 
                                int ,
                                int , 
                                int 
                               );
  int classifySample(Sample &, std::vector<double> &); // for forest 
  int classifySample(Sample &); // for a single tree
};

#endif // TREE_H