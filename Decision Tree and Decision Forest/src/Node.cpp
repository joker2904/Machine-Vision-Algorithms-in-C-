#include "Node.h"
#include <iostream>
using namespace std;

bool Node::NodeTrain(std::vector<Sample>* SampleSet,
                     std::vector<BinaryTest>* TestSet,
                     int maximumDepthOfTree,
                     int MinimumPatchesAtLeafNode
                    )
{
    if (this->CurrentDepthofNode >= maximumDepthOfTree || (SampleSet->size() < (2 * MinimumPatchesAtLeafNode)) || this->Classified(SampleSet)) 
    {
        //should be leaf node;
        this->SamplePatches = *SampleSet;
        this->isLeaf = true;
        (this->ProbabilitiesPerClass).resize(this->NoOfClasses);
        // Calculate probabilities of each class and store it
        for (int index = 0; index < (this->SamplePatches).size(); ++index) 
        {
          (this->ProbabilitiesPerClass)[ (this->SamplePatches)[index].label ] += ( 1.0/(this->SamplePatches).size() ); 
        }
        return true;
    }
    
    /* Algorithm Explain :
       1. Split the sample set contained in this node , wrt every binary test.
       2. Calculate the information gain based on the current splitting.
       3. After all the splits have been tested and the split based on the best InformationGain has been found, split the sample set according to this best IG.
       4. After the final splitting has been done, the call the same algorithm recursively on the 2 child nodes.    
    */
    
    // Splitting this current sample set contained in this node, based on the highest Information Gain 
    double bestInformationGain = -std::numeric_limits<float>::infinity();
    BinaryTest bestTestCase = (*TestSet)[0];
    std::vector<Sample> OptimalLeftPartition;
    std::vector<Sample> OptimalRightPartition;
    for (int index = 0; index < TestSet->size(); ++index)
    {
        std::vector<Sample> LeftPartition;
        std::vector<Sample> RightPartition;
        this->PartionSampleSetBasedOnBinaryTest( &((*TestSet)[index]),SampleSet,&LeftPartition,&RightPartition); // Current Split, based on this TestSet
        double InformationGain = this->CalculateInformationGain(SampleSet,&LeftPartition,&RightPartition);
        if(InformationGain > bestInformationGain) {
            bestTestCase = (*TestSet)[index];
            bestInformationGain = InformationGain;
            OptimalLeftPartition  = LeftPartition;
            OptimalRightPartition = RightPartition;
        }
    }
    //Set the sample parameters to that of this node
    this->channel    = bestTestCase.c;
    this->col        = bestTestCase.x;
    this->row        = bestTestCase.y;
    this->threshold  = bestTestCase.thresh;
    
    // Create 2 new children and pass on the 2 split(or partioned samples to them respectively */
    this->leftChild  = new Node(this->CurrentDepthofNode + 1, this->NoOfClasses);
    this->rightChild = new Node(this->CurrentDepthofNode + 1, this->NoOfClasses);
    // Recursively train the children nodes with the partioned samples respectively 
    this->leftChild->NodeTrain(&OptimalLeftPartition, TestSet, maximumDepthOfTree, MinimumPatchesAtLeafNode);
    this->rightChild->NodeTrain(&OptimalRightPartition, TestSet, maximumDepthOfTree, MinimumPatchesAtLeafNode);
    return true;
}

bool Node::Classified(std::vector<Sample>* SampleSet)
{
   std::vector<int> classUnique(this->NoOfClasses);
    for (int index = 0; index < SampleSet->size(); ++index)
    {
      classUnique[ ( (*SampleSet)[index] ).label ] = 1;
    }
    int sum = 0;
    for (std::vector<int>::iterator itr = classUnique.begin(); itr != classUnique.end(); itr++ )
        sum += *itr ;
    return (sum == 1);   
}

double Node::CalculateEntropyBasedOnSplit(std::vector<Sample>* Samples)
{
  double Entropy = 0.0;
  std::vector<double> samplesPerClass(this->NoOfClasses);
  // Calculate Number of samples in each class
  for (int index = 0; index < Samples->size(); ++index) 
  {
    samplesPerClass[ (*Samples)[index].label ] ++; 
  }
 
  for (int index = 0; index < samplesPerClass.size(); ++index) 
  {
      double probabilityOfClass = samplesPerClass[index]/(Samples->size());
      if(probabilityOfClass < 0.00001) 
          continue;
      Entropy +=  (probabilityOfClass * log2(probabilityOfClass));
  }
  return (-0.1d * Entropy);
}

double Node::CalculateInformationGain(std::vector<Sample>* SampleSet, 
                                      std::vector<Sample>* LeftSplitSampleSet, 
                                      std::vector<Sample>* RightSplitSampleSet 
                                     )
{
    if(LeftSplitSampleSet->empty() || RightSplitSampleSet->empty())
        return -std::numeric_limits<float>::infinity();
    double HS = this->CalculateEntropyBasedOnSplit(SampleSet);
    double HR =  (double)(LeftSplitSampleSet->size())  * this->CalculateEntropyBasedOnSplit(LeftSplitSampleSet);
    double HL =  (double)(RightSplitSampleSet->size()) * this->CalculateEntropyBasedOnSplit(RightSplitSampleSet);
    return HS - ((HR + HL)/SampleSet->size());
}

void Node::PartionSampleSetBasedOnBinaryTest(BinaryTest* TestInstance,
                                             std::vector<Sample>* SampleSet,
                                             std::vector<Sample>* LeftPartition,
                                             std::vector<Sample>* RightPartition
                                            )
{  /*
    for (int index = 0; index < SampleSet->size(); ++index)
    {
        if (this->SampleSplitCriteria( TestInstance, &((*SampleSet)[index]) ))
            LeftPartition->push_back((*SampleSet)[index]);
        else
            RightPartition->push_back((*SampleSet)[index]);
    }*/
    
    for( std::vector<Sample>::iterator itr = SampleSet->begin(); itr != SampleSet->end(); ++itr )
    {
        if (this->SampleSplitCriteria( *TestInstance, *itr) )
            LeftPartition->push_back(*itr);
        else
            RightPartition->push_back(*itr);
    }
        
}

bool Node::SampleSplitCriteria(BinaryTest TestInstance, Sample SampleInstance)
{
   
   return ( (SampleInstance.Image->at<cv::Vec3b>( SampleInstance.bbox.y + TestInstance.y, SampleInstance.bbox.x + TestInstance.x)[TestInstance.c]) <  TestInstance.thresh );
}


std::vector<double> Node::classifySample(Sample s) 
{
    if (rightChild == NULL || leftChild == NULL) 
    {
       return this->ProbabilitiesPerClass;
    }
    
    BinaryTest t;
    t.c =      this->channel;
    t.x =      this->col;
    t.y =      this->row;
    t.thresh = this->threshold;
    
    if (this->SampleSplitCriteria(t,s))
        return this->rightChild->classifySample(s);
    else
        return this->leftChild->classifySample(s);
}











