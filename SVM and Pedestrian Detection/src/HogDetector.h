#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <stdio.h>
#include <vector>
#include <map>
#include <algorithm>


using namespace std;
using namespace cv;

//extern void PredictValuesBasedOnC(cv::Ptr<CvSVM> ,float ,float& , float& );
extern void drawBoundingBox(string DisplayName,Mat& image,vector<Rect>& found_filtered);


class myHogDetector
{
  private:
    cv::Ptr<CvSVM> TrainedSVM;
   
    int lengthHogDescriptor;
    Rect BR;
  public:
       
    myHogDetector()
    {
       lengthHogDescriptor = 3780;
       
    }
    
    ~myHogDetector()
    {}
    
    bool setSVMDetector(cv::Ptr<CvSVM> c)
    {
       TrainedSVM = c; 
    }
    
    bool mDetectMultiple(Mat&,vector<Rect>&);
    bool Test(Mat&,vector<Rect>&);
    bool Prediction(Mat&,float&,float&);
    float MaximumSuppressionArea(float** , 
                                 float** ,
                                 std::map< float,std::pair<int,int> >&,
                                 std::map< std::pair<int,int> , Rect>&,
                                 vector<Rect>&
                                );
    bool ScanScaledImage( Mat& image, 
                                     float** LabelMatrix,
                                     float** PredictionMatrix,
                                     std::map< float, std::pair<int,int> >& PredictionLabelMap,
                                     std::map< std::pair<int,int> , Rect >& DetectionMap,
                                     std::map< std::pair<int,int>, float >& PredictionMap
                                   );
    float** Allocate(int,int); 
    
    void EliminateOverlappingRectangles( int,int,float** LabelMatrix,
                                     float** PredictionMatrix,
                                     std::map< float, std::pair<int,int> >& PredictionLabelMap,
                                     std::map< std::pair<int,int> , Rect >& DetectionMap
                                   );
    void  EliminateOverlappingRectangles( std::map< float, std::pair<int,int> >& PredictionLabelMap,
                                                     std::map< std::pair<int,int> , Rect >& DetectionMap,
                                                     std::map< std::pair<int,int> , float>& PredictionMap,
                                                     std::vector<Rect>& Detections,
                                                      Mat& image
                                                   );
};