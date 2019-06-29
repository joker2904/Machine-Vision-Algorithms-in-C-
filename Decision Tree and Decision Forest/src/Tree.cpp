#include "Tree.h"
#include <iostream>
using namespace std;
  
void Tree::setParam(TreeParam* value)
{
    if(root != NULL)
        delete root;
    root = new Node(0, value->numOfClasses);

    colorClasses.resize(value->numOfClasses);
    if(value->numOfClasses == 4) {
        colorClasses[0] = 0; //black
        colorClasses[1] = cv::Scalar(0, 0, 255); // red
        colorClasses[2] = cv::Scalar(255, 0, 0); // blue
        colorClasses[3] = cv::Scalar(0, 255, 0); // green

    }   
    else
    {
       cv::RNG randomNumber(time(NULL));
       for (int i = 0; i < colorClasses.size(); ++i)
       {
           cv::Scalar color = cv::Scalar(randomNumber.uniform(0, 255), randomNumber.uniform(0, 255), randomNumber.uniform(0, 255));
           colorClasses[i] = color;
       }
    }

    
}


void Clear(Node* root)
{
  if(root)
  {
    Clear(root->leftChild);
    Clear(root->rightChild);
    delete root;
  }
}

Tree::~Tree()
{
  Node* temp = this->root;
  //Clear(temp);    
}

 
std::vector<BinaryTest> Tree::GenerateRandomBinaryTests() 
{
    cv::RNG randomNumber(time(NULL));
    const int NoOfchannels = 10;
    const int NoOfPixelLocations = 100;
    const int NoOfThresh = 50;

    std::vector<BinaryTest> RandomBinaryTests; 
    int index = 0;
    for (int channel = 0; channel < NoOfchannels; ++channel)
    {
        int c = randomNumber.uniform((int)0, (int)4);
        for (int pixelLoc = 0; pixelLoc < NoOfPixelLocations; ++pixelLoc) 
        {
            int x = randomNumber.uniform((int)0, (int) param.ImagePatchDimensions);
            int y = randomNumber.uniform((int)0, (int) param.ImagePatchDimensions);
            for (int thres = 0; thres < NoOfThresh; ++thres) 
            {
                BinaryTest RandomTest;
                RandomTest.c = c;
                RandomTest.x = x;
                RandomTest.y = y;
                RandomTest.thresh = randomNumber.uniform((int)0, (int)255);
                RandomBinaryTests.push_back(RandomTest);
            }
        }
    }
    
    return RandomBinaryTests;
}
        
        
        
bool SampleGenerationCondition(int ImagePatchesPerClass, std::vector<int> NoOfClassInSample)
{
   for( std::vector<int>::iterator itr = NoOfClassInSample.begin(); itr !=  NoOfClassInSample.end(); ++itr )
       if( *itr < ImagePatchesPerClass )
           return false;
   return true;   
}

void Tree::ConstructTrainingSamples(std::vector<cv::Mat>* trainingImgs,
                         std::vector<cv::Mat>* correspondingTruthSegments,
                         std::vector<Sample>*  samplePatchesPerClass, 
                         int NoOfClasses ,
                         int ImagePatchesPerClass , 
                         int ImagePatchDimensions 
                        )
{
      
    cv::RNG randomNumberGenerator(time(NULL));

    int xLowerBound = 0, yLowerBound = 0;
    int xHigherBound = ((*trainingImgs)[0].cols - ImagePatchDimensions + 1);
    int yHigherBound = ((*trainingImgs)[0].rows - ImagePatchDimensions + 1);
    int ImageCentroid = ImagePatchDimensions / 2.0f;
    std::vector<int> NoOfClassInSample(NoOfClasses);
    
   
    while( !SampleGenerationCondition(ImagePatchesPerClass,NoOfClassInSample) )    
    {
        //get new x and y for the bbox
        int x = randomNumberGenerator.uniform(xLowerBound, xHigherBound);
        int y = randomNumberGenerator.uniform(yLowerBound, yHigherBound);

        // get random image
        int imageIndex = randomNumberGenerator.uniform(0, trainingImgs->size());
        cv::Mat sampleMatrix = (*trainingImgs)[imageIndex];

        // pixel to be tested
        int row = y + ImageCentroid;
        int col = x + ImageCentroid;

        // ground truth of the pixel
        int truthSegmentLabel = (*correspondingTruthSegments)[imageIndex].at<cv::Vec3b>(row, col)[0];

        samplePatchesPerClass->push_back( Sample(new cv::Mat(sampleMatrix), cv::Rect(x, y, ImagePatchDimensions, ImagePatchDimensions), truthSegmentLabel));
        NoOfClassInSample[truthSegmentLabel]++;
    }
   
}        
        
        
        
        
 void Tree::Train(std::vector<cv::Mat>* trainImgs, std::vector<cv::Mat>* trainSegMaps, int numClasses)
 {

    std::vector<Sample> samplePatchesPerClass;
    this->ConstructTrainingSamples(trainImgs, trainSegMaps, &samplePatchesPerClass, numClasses, param.minTrainPatchesPerClass, param.ImagePatchDimensions);
    std::vector<BinaryTest> binaryTests = this->GenerateRandomBinaryTests();
    this->isTreeTrained = this->root->NodeTrain(&samplePatchesPerClass, &binaryTests, param.depthOfTree, param.minImagePatchesAtLeaf);
 }
 
 
 bool Tree::isTrained(void)
 {
     return isTreeTrained;
 }
  
 
 
void Tree::testImage(cv::Mat& testImg, cv::Mat& segMapOut)
{
    segMapOut.create(testImg.rows, testImg.cols, testImg.type());
    for (int col = 0; col < (testImg.cols - param.ImagePatchDimensions); ++col)
    {
        for (int row = 0; row < (testImg.rows - param.ImagePatchDimensions); ++row)
        {
            cv::Rect bbox(col, row, param.ImagePatchDimensions, param.ImagePatchDimensions);
            Sample s(&testImg, bbox, -1);
            int classIdx = classifySample(s);
            int x =  col + param.ImagePatchDimensions / 2.0f;
            int y =  row + param.ImagePatchDimensions / 2.0f;
            segMapOut.at<cv::Vec3b>(y, x)[0] = colorClasses[classIdx][0];
            segMapOut.at<cv::Vec3b>(y, x)[1] = colorClasses[classIdx][1];
            segMapOut.at<cv::Vec3b>(y, x)[2] = colorClasses[classIdx][2];

        }
    }
}

int Tree::classifySample(Sample& s)
{
    std::vector<double> probs = root->classifySample(s);

    int classNum = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
    return classNum;
}

int Tree::classifySample(Sample& s, std::vector<double>& probs) 
{
    std::vector<double> probs_ = root->classifySample(s);
    for (int var = 0; var < probs.size(); ++var) {
        probs[var] += probs_[var];
    }
    int classNum = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
    return classNum;
}

 
 