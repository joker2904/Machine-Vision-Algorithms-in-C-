#ifndef NODE_H
#define NODE_H
#include <Sample.h>
#include <BinaryTest.h>
/// Node of a tree
class Node
{
 public:
   Node* leftChild;
   Node* rightChild;
   int CurrentDepthofNode;
   int NoOfClasses;
   bool isLeaf;
   std::vector<Sample> ContainedSamplePatches; // The container of samples which are carried by this node of the tree
   int channel;
   int col;      
   int row;        
   double threshold;  
   std::vector<double> ProbabilitiesPerClass;
   std::vector<Sample> SamplePatches;
   
   Node(int currentDepth,int NoOfClasses)
   {
     this->CurrentDepthofNode = currentDepth;
     this->NoOfClasses = NoOfClasses;
     this->isLeaf = false; 
     leftChild = NULL;
     rightChild = NULL;
   }
      
 
   bool NodeTrain(std::vector<Sample>*,
                        std::vector<BinaryTest>*,
                        int ,
                        int 
                 );
   
   double CalculateEntropyBasedOnSplit(std::vector<Sample>*);
 
   double CalculateInformationGain(std::vector<Sample>* , 
                                   std::vector<Sample>* , 
                                   std::vector<Sample>*  
                                  );
   
   void PartionSampleSetBasedOnBinaryTest(BinaryTest* ,
                                          std::vector<Sample>* ,
                                          std::vector<Sample>* ,
                                          std::vector<Sample>* 
                                         );
   
   bool SampleSplitCriteria(BinaryTest,Sample);
   
   bool Classified(std::vector<Sample>*);
   std::vector<double> classifySample(Sample);
};

#endif // NODE_H


//// Functions to compute the splitting decison and entropy 