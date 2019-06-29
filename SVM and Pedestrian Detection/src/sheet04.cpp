#include <iostream>
#include <fstream>
#include <time.h> // for random number generation
#include <iomanip> // set precision of output string
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <stdio.h>
using namespace std;
using namespace cv;


#include "HogDetector.h" 

/// Global variables
int x=16, y=16;
int width = 64, height = 128;
int lengthHogDescriptor = 3780;
int numSamples = 10;


void filterNumOfDetections(vector<Rect>& found,vector<Rect>& found_filtered);/// functions for section 1

void drawBoundingBox(string DisplayName,Mat& image,vector<Rect>& found_filtered)
{
  cv::namedWindow( DisplayName, cv::WINDOW_AUTOSIZE);
  //std::sleep(1);
  for ( vector<Rect>::iterator itr = found_filtered.begin() ; itr != found_filtered.end(); ++itr )
  {
     cv::rectangle(image,*itr, 0, 3);     
  }
  cv::imshow(DisplayName, image);
  cv::waitKey(0);   
}

void saveMatAsciiWithHeader(const string& filename, Mat& matData);  /// functions for section 2
void readMatAsciiWithHeader(const string& filename, Mat& matData); /// functions for section 3
void readFilenamesFromAscii(string filename, vector<string>& vecFilelist); // Helper Functions

void ex1()
{
    cout << "Section 1 - OpenCV HOG :: Default People Detector" << endl;
    
    /// Load images
    string path = "../section_1_testImages/";
    string filelist = path + "filenames.txt";
    vector<string> vecFilelist;
    readFilenamesFromAscii(filelist, vecFilelist);

    /// TODO: Create HOG feature extractor object
    // ...
 
    HOGDescriptor hdt;
    hdt.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    
    /// Create output containers
    int numImgs = (int) vecFilelist.size();
    string filename;


    for ( int i = 0; i < numImgs; i++ )
    {
        filename = path + vecFilelist[i];
        Mat image = imread(filename);
        if (image.data)
            cout<<"\nImage " << i+1 << " loaded successfully"<<endl;
        else
            break;
        /// TODO: Apply HOG detector
	// ...
        vector<Rect> BB ;        
        //hdt.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
        hdt.detectMultiScale(image,BB);
        cout<<"\nDefault Detector Image < No of Bounding Rectangles found - "<<BB.size()<<" >";
        drawBoundingBox("Default Detector",image,BB);
  
    }

    cout << "\nDone" << endl;
    destroyAllWindows();
    waitKey(0);
}

void ex11()
{
    cout << "Section 1 - OpenCV HOG" << endl;
    
    /// Load images
    string path = "../section_1_testImages/";
    string filelist = path + "filenames.txt";
    vector<string> vecFilelist;
    readFilenamesFromAscii(filelist, vecFilelist);

    /// TODO: Create HOG feature extractor object
    // ...
    HOGDescriptor hdt_dialmer( Size(48, 96), Size(16, 16), Size(8, 8), Size(8, 8), 9);
    hdt_dialmer.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());
    
    /// Create output containers
    int numImgs = (int) vecFilelist.size();
    string filename;


    for ( int i = 0; i < numImgs; i++ )
    {
        filename = path + vecFilelist[i];
        Mat image = imread(filename);
        if (image.data)
            cout<<"\nImage " << i+1 << " loaded successfully"<<endl;
        else
            break;
        /// TODO: Apply HOG detector
	// ...
                 
        vector<Rect> BBD;      
        hdt_dialmer.detectMultiScale(image,BBD, 0.5, Size(8,8), Size(32,32), 1.05, 2, true);
        cout<<"\nDiamler Detector Image < No of Bounding Rectangles found - "<<BBD.size()<<" >";
        drawBoundingBox("Diamler Detector",image,BBD);
    }

    cout << "\nDone" << endl;
    destroyAllWindows();
    waitKey(0);
}



void ex2()
{
    cout << "Section 2 - Extract HOG features" << endl;

    /// Load image names
    string path_train = "../section_2-3_data/train/";
    string filelist_trainPos = path_train + "filenamesTrainPos.txt";
    string filelist_trainNeg = path_train + "filenamesTrainNeg.txt";
    vector<string> vecFilelist_trainPos; 
    readFilenamesFromAscii(filelist_trainPos, vecFilelist_trainPos);
    vector<string> vecFilelist_trainNeg;
    readFilenamesFromAscii(filelist_trainNeg, vecFilelist_trainNeg);
    
    HOGDescriptor hog;
    Mat matROI;
    vector<float> descriptors;
    int counter, numImgsPos, numImgsNeg;
    uint64 initValue = time(0);
    RNG rng(initValue);
    
    /// Train Imgs
    counter = 0;
    numImgsPos = (int) vecFilelist_trainPos.size();
    numImgsNeg = (int) vecFilelist_trainNeg.size();
    Mat matDescriptorsTrain;//(numImgsPos + numSamples * numImgsNeg,lengthHogDescriptor,CV_32F);
    Mat matLabelsTrainPos(numImgsPos, 1, CV_32FC1, Scalar(1));
    Mat matLabelsTrainNeg(numSamples * numImgsNeg, 1, CV_32FC1, Scalar(-1));
    Mat matLabelsTrain;
    vconcat(matLabelsTrainPos,matLabelsTrainNeg,matLabelsTrain);

    /// Pos
    Mat PosTrainingDescriptors;
    for ( int i = 0; i < numImgsPos; i++ )
    {
        string filename = path_train + "pos/" + vecFilelist_trainPos[i];
	
	/// TODO: Load image
        // ...
        Mat image = imread(filename);
        cout<<endl<<filename<<"         "<<image.rows<<" , "<<image.cols;
	
	/// TODO: Crop image to suitable size ( get a crop of size 64 x 128 from the center) 
	// ...
        image = image( cv::Rect(image.cols/2 - 31 , image.rows/2 - 63 , 64,128) & cv::Rect(0, 0, image.cols, image.rows) ); // using a rectangle

        /// TODO: Extract HOG feature descriptor
	// ...
        //descriptors.resize(lengthHogDescriptor);
       
        std::vector<Point> locations ;//= std::vector< Point >()  
        locations.push_back( Point(32 , 64) );
        hog.compute(image, descriptors ); //,Size(64,128),Size(4,4)); //,locations);
	
	//for( int j = 0; j< descriptors.size(); ++j)
        //    cout<<descriptors[j];
        //cout<< "\n Size == "<<descriptors.size()<<" , "<<descriptors.size()/lengthHogDescriptor<<"   "<<counter;
        //cout<<"\nSize="<<descriptors.size()<<" ::"<<hog.getDescriptorSize()<<" :: "<<counter<<"::: //"<<PosTrainingDescriptors.rows<<","<<PosTrainingDescriptors.cols;
        
        cv::Mat temp = cv::Mat(1, lengthHogDescriptor, CV_32F, &descriptors.front()) ;
        //cout<<temp;
        PosTrainingDescriptors.push_back(temp);
        /// Count number of descriptors
        counter++;
    }

     Mat NegTrainingDescriptors;
    /// Neg
    for ( int i = 0; i < numImgsNeg; i++ )
    {
        string filename = path_train + "neg/" + vecFilelist_trainNeg[i];
        Mat image = imread(filename);
        cout<<endl<<filename<<"         "<<image.rows<<" , "<<image.cols;
        /// TODO: Sample 10 descriptors at random
	for (int n = 0; n < numSamples; n++)
        {
	    y = rng.uniform(0, image.rows - height);
	    x = rng.uniform(0, image.cols - width);
	    // ...
	    
	    /// TODO: Extract HOG feature descriptor
	    // ...
	    Mat Nimage = image( cv::Rect(x, y , 64,128) & cv::Rect(0, 0, image.cols, image.rows) ); // using a rectangle
        
            //std::vector<Point> locations ;//= std::vector< Point >()  
            //locations.push_back( Point(x , y) );
            hog.compute(Nimage, descriptors); // ,Size(64,128),Size(8,8)); //,locations);
            //cout<< "\n Size Negative == "<<descriptors.size()<<" , "<<descriptors.size()/lengthHogDescriptor;
             
            cv::Mat temp = cv::Mat(1, lengthHogDescriptor, CV_32F, &descriptors.front()) ;
            NegTrainingDescriptors.push_back(temp);
            //cout<<temp;
            counter++;
	   
	}
        
    }
    
    vconcat(PosTrainingDescriptors,NegTrainingDescriptors,matDescriptorsTrain);
    cout<<"\n Dimensions :::\n"<<PosTrainingDescriptors.rows<<","<<PosTrainingDescriptors.cols<<" :: "<<NegTrainingDescriptors.rows<<","<<NegTrainingDescriptors.cols<<" :: "<<matDescriptorsTrain.rows<<" ,"<<matDescriptorsTrain.cols;

    /// Store descriptors
    saveMatAsciiWithHeader("../TrainHogDescs.dat", matDescriptorsTrain);
    saveMatAsciiWithHeader("../matLabelsTrain.dat", matLabelsTrain);

    cout << "Done" << endl;
    waitKey(0);
    destroyAllWindows();
}





void ex3()
{
    cout << "Exercise 3 - Train SVM and predict confidence values" << endl;
    /// Train SVMs
    Mat matDescTrain, matDescTest;
    Mat labelsTrain, labelsTest;
    readMatAsciiWithHeader("../TrainHogDescs.dat", matDescTrain);
    readMatAsciiWithHeader("../matLabelsTrain.dat", labelsTrain);

    /// TODO: Set up opencv SVM
    // ...
    CvSVM c1,c2,c3;
    /// TODO: Create 3 SVMs with different C values, train them with the training data and save them
    
    // ... Setting the parameters for the 3 SVMs , according to the 3 values of C
    CvSVMParams param1;//(CvSVM::C_SVC, CvSVM::LINEAR, 0 ,0 ,0 ,0.01d ,0 ,0 ,0 ,0);
    param1.svm_type     = CvSVM::C_SVC;
    param1.C            = 0.01;
    param1.kernel_type  = SVM::LINEAR;
    param1.term_crit    = TermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

    CvSVMParams param2;//(CvSVM::C_SVC, CvSVM::LINEAR, 0,0,0,1.00,0,0,0,0);
    param2.svm_type     = CvSVM::C_SVC;
    param2.C            = 1.0;
    param2.kernel_type  = SVM::LINEAR;
    param2.term_crit    = TermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

    CvSVMParams param3;//(CvSVM::C_SVC, CvSVM::LINEAR, 0,0,0,100.0,0,0,0,0);   
    param3.svm_type     = CvSVM::C_SVC;
    param3.C            = 100.0;
    param3.kernel_type  = SVM::LINEAR;
    param3.term_crit    = TermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

    
    // Training the 3 SVMS with the training data collected above
    cout<<"\n Training SVM 1::";
    c1.train(matDescTrain, labelsTrain, Mat(), Mat(), param1 );
    for( int i = 0; i < matDescTrain.rows; ++i )
    {
      float pred1 = c1.predict(matDescTrain.row(i),true);
      cout<<"\n"<<pred1<<","<<labelsTrain.row(i)<<" , Trainging Label ="<<c1.predict(matDescTrain.row(i),false);
    }
    
    cout<<"\n Training SVM 2::";
    c2.train(matDescTrain, labelsTrain, Mat(), Mat(), param2 );
    for( int i = 0; i < matDescTrain.rows; ++i )
    {
      float pred1 = c2.predict(matDescTrain.row(i),true);
      cout<<"\n"<<pred1<<","<<labelsTrain.row(i)<<" , Trainging Label ="<<c2.predict(matDescTrain.row(i),false);
    }
    
    cout<<"\n Training SVM 3::";
    c3.train(matDescTrain, labelsTrain, Mat(), Mat(), param3 );
    for( int i = 0; i < matDescTrain.rows; ++i )
    {
      float pred1 = c3.predict(matDescTrain.row(i),true);
      cout<<"\n"<<pred1<<","<<labelsTrain.row(i)<<" , Trainging Label ="<<c3.predict(matDescTrain.row(i),false);
    }
    

    /// Classify test images
    string path_test = "../section_2-3_data/test/";
    string filelist_testPos = path_test + "filenamesTestPos.txt";
    string filelist_testNeg = path_test + "filenamesTestNeg.txt";
    vector<string> filenames_testPos;
    readFilenamesFromAscii(filelist_testPos, filenames_testPos);
    vector<string> filenames_testNeg;
    readFilenamesFromAscii(filelist_testNeg, filenames_testNeg);

    // TODO: Create HOG object
    // ...

    // TODO: Classify positive test images and save results in an ascii file
    // ...

    // TODO: Classify negative test images and save results in an ascii file
    // ...
    
    
    cout<<"\n ::::::::::::: Starting to extract the HOG feature for each test image ::::::::::::::::::::::::::::::::::::";
    HOGDescriptor hog;
    Mat matROI;
    vector<float> descriptors;
    int counter, numImgsPos, numImgsNeg;
    uint64 initValue = time(0);
    RNG rng(initValue);
    
    /// Train Imgs
    counter = 0;
    numImgsPos = (int) filenames_testPos.size();
    numImgsNeg = (int) filenames_testNeg.size();
    Mat matDescriptorsTest;//(numImgsPos + numSamples * numImgsNeg,lengthHogDescriptor,CV_32F);
    Mat matLabelsTestPos(numImgsPos, 1, CV_32FC1, Scalar(1));
    Mat matLabelsTestNeg(numSamples * numImgsNeg, 1, CV_32FC1, Scalar(-1));
    Mat matLabelsTest;
    vconcat(matLabelsTestPos,matLabelsTestNeg,matLabelsTest);
    
    cout<<"\n ::::::::::: reading positive images :::::::::::::::::::::::::";
    /// Pos
    Mat PosTestDescriptors;
    for ( int i = 0; i < numImgsPos; i++ )
    {
        string filename = path_test + "pos/" + filenames_testPos[i];
	
	/// TODO: Load image
        // ...
        Mat image = imread(filename);
	
	/// TODO: Crop image to suitable size ( get a crop of size 64 x 128 from the center) 
	// ...
      //  image = image( cv::Rect(image.cols/2 - 31 , image.rows/2 - 63 , 64,128) & cv::Rect(0, 0, image.cols, image.rows) ); // using a rectangle

        /// TODO: Extract HOG feature descriptor
	// ...
        //descriptors.resize(lengthHogDescriptor);
       
        std::vector<Point> locations ;//= std::vector< Point >()  
        locations.push_back( Point(32 , 64) );
        hog.compute(image, descriptors ); //,Size(64,128),Size(4,4)); //,locations);
	
	//for( int j = 0; j< descriptors.size(); ++j)
        //    cout<<descriptors[j];
        //cout<< "\n Size == "<<descriptors.size()<<" , "<<descriptors.size()/lengthHogDescriptor<<"   "<<counter;
        //cout<<"\nSize="<<descriptors.size()<<" ::"<<hog.getDescriptorSize()<<" :: "<<counter<<"::: //"<<PosTrainingDescriptors.rows<<","<<PosTrainingDescriptors.cols;
        cout<<"\n Positive test Image No "<<counter+1<<" :: Length of HOG descriptor - "<<descriptors.size();
        cv::Mat temp = cv::Mat(1, lengthHogDescriptor, CV_32F, &descriptors.front()) ;
        //cout<<temp;
        PosTestDescriptors.push_back(temp);
        /// Count number of descriptors
        counter++;
    }

     cout<<"\n ::::::::::: reading negative images :::::::::::::::::::::::::";
    Mat NegTestDescriptors;
    /// Neg
    for ( int i = 0; i < numImgsNeg; i++ )
    {
        string filename = path_test + "neg/" + filenames_testNeg[i];
        Mat image = imread(filename);

        /// TODO: Sample 10 descriptors at random
	for (int n = 0; n < numSamples; n++)
        {
	    y = rng.uniform(0, image.rows - height);
	    x = rng.uniform(0, image.cols - width);
	    // ...
	    
	    /// TODO: Extract HOG feature descriptor
	    // ...
	    Mat Nimage = image( cv::Rect(x, y , 64,128) & cv::Rect(0, 0, image.cols, image.rows) ); // using a rectangle
        
            //std::vector<Point> locations ;//= std::vector< Point >()  
            //locations.push_back( Point(x , y) );
            hog.compute(Nimage, descriptors); // ,Size(64,128),Size(8,8)); //,locations);
            //cout<< "\n Size Negative == "<<descriptors.size()<<" , "<<descriptors.size()/lengthHogDescriptor;
            cout<<"\n Negative test Image No "<<counter+1<<" :: Length of HOG descriptor - "<<descriptors.size();
            cv::Mat temp = cv::Mat(1, lengthHogDescriptor, CV_32F, &descriptors.front()) ;
            NegTestDescriptors.push_back(temp);
            //cout<<temp;
            counter++;
	   
	}
        
    }
    
    vconcat(PosTestDescriptors,NegTestDescriptors,matDescriptorsTest);
    //cout<<"\n Dimensions :::\n"<<PosTrainingDescriptors.rows<<","<<PosTrainingDescriptors.cols<<" :: "<<NegTrainingDescriptors.rows<<","<<NegTrainingDescriptors.cols<<" :: "<<matDescriptorsTrain.rows<<" ,"<<matDescriptorsTrain.cols;

    /// Store descriptors
    saveMatAsciiWithHeader("../TestHogDescs.dat", matDescriptorsTest);
    saveMatAsciiWithHeader("../matLabelsTest.dat", matLabelsTest);
    
    // Test the features with the 3 different SVMs created 
    Mat l1,l2,l3;
    float error;
    
    c1.predict(matDescriptorsTest,l1);
    
    //cv::bitwise_xor(matLabelsTest, l1, dst);        
    //error = cv::countNonZero(dst);
    cout<<"\n Labels from SVM1 :::";
    error = 0.0;
    for( int i = 0; i < l1.rows; ++i )
    {
        cout<<"\n"<<matLabelsTest.at<float>(i,0)<<","<<l1.at<float>(i,0);
        if ( matLabelsTest.at<float>(i,0) != l1.at<float>(i,0)  )
            error++;
    }
    cout<<"\n No of errors from SVM1 = "<<error;    
        
    
    c2.predict(matDescriptorsTest,l2);
    cout<<"\n Labels from SVM2 :::";
    error = 0.0;
    for( int i = 0; i < l2.rows; ++i )
    {
        cout<<"\n"<<matLabelsTest.at<float>(i,0)<<","<<l2.at<float>(i,0);
        if ( matLabelsTest.at<float>(i,0) != l2.at<float>(i,0)  )
            error++;
    } 
    //cv::bitwise_xor(matLabelsTest, l1, dst);        
    //error = cv::countNonZero(dst);
    cout<<"\n No of errors from SVM2 = "<<error;   
    
    c3.predict(matDescriptorsTest,l3);
    error = 0.0;
    for( int i = 0; i < l3.rows; ++i )
    {
        cout<<"\n"<<matLabelsTest.at<float>(i,0)<<","<<l3.at<float>(i,0);
        if ( matLabelsTest.at<float>(i,0) != l3.at<float>(i,0)  )
            error++;
    }
    //cv::bitwise_xor(matLabelsTest, l1, dst);        
    //error = cv::countNonZero(dst);
    cout<<"\n No of errors from SVM3 = "<<error;   
    
    /*
    cout<<"\n Labels from SVM1 ::"<<l1;
    cout<<"\n Labels from SVM2 ::"<<l2;
    cout<<"\n Labels from SVM3 ::"<<l3;
    */
    cout << "Done" << endl;
    
}


Ptr<CvSVM> PredictValuesBasedOnC(float C,float& precision, float& recall)
{
    cout << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~";
    cout << "\nTrain SVM with threshold value ::"<<C<<endl;
    /// Train SVMs
    Mat matDescTrain, matDescTest;
    Mat labelsTrain, labelsTest;
    readMatAsciiWithHeader("../TrainHogDescs.dat", matDescTrain);
    readMatAsciiWithHeader("../matLabelsTrain.dat", labelsTrain);


    Ptr<CvSVM> c1 = new CvSVM();
    // ... Setting the parameters for the SVM , according to the value of C
    CvSVMParams param1;
    param1.svm_type     = CvSVM::C_SVC;
    param1.C            = C;
    param1.kernel_type  = SVM::LINEAR;
    param1.term_crit    = TermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

     
    // Training the SVM with the training data collected above 
    float totalprecision = 0.0,trainingerror=0.0;
    c1->train(matDescTrain, labelsTrain, Mat(), Mat(), param1 );
    ofstream oStream("../PredictionScores/PredictionScoresForC_"+to_string(C)+".dat");
    oStream << matDescTrain.rows << " " << 2 << endl;
    for( int i = 0; i < matDescTrain.rows; ++i )
    {
      
      float PredictionScore = c1->predict(matDescTrain.row(i),true);
      float PredictionLabel = c1->predict(matDescTrain.row(i),false);
      totalprecision += PredictionScore;
      oStream << PredictionScore << " " << PredictionLabel << endl;
      if( labelsTrain.at<float>(i,0) != c1->predict(matDescTrain.row(i),false) )
          trainingerror++;
    }
  
    oStream.close();
    //Getting the testdata which was created before
    readMatAsciiWithHeader("../TestHogDescs.dat", matDescTest);
    readMatAsciiWithHeader("../matLabelsTest.dat", labelsTest);

    // Test the features with the 3 different SVMs created 
    Mat labels;
    int error=0,tp=0,tn=0,fp=0,fn=0;
    c1->predict(matDescTest,labels);
    for( int i = 0; i < labels.rows; ++i )
    {
        int testLabel = labelsTest.at<float>(i,0) > 0 ? 1:0;
        int predictionlabel = labels.at<float>(i,0) > 0 ? 1:0;
        if ( testLabel != predictionlabel  )
            error++;
        if( testLabel == 0 )
        {
          if(predictionlabel == 0)
             tn ++;
          if(predictionlabel == 1)
             fp ++;            
        }
        if( testLabel == 1 )
        {
          if(predictionlabel == 0)
             fn ++;
          if(predictionlabel == 1)
             tp ++; 
        }
        precision = (float)( (float)tp / (float)( fp + tp ) );
        recall = (float) ((float)tp / (float)( fn + tp )) ;
    }
    cout<<"\n C = "<<C;
    cout<<"\n Testing Error = "<<(float)(error)/(float)(labels.rows);
    cout<<"\n Training Error = "<<(float)(trainingerror)/(float)(matDescTrain.rows);
    cout<<"\n Total Sum of Precision ="<<totalprecision<<" (Precision is the distance of each training sample from the margin)";   
    cout<<"\n SVM address = "<<c1;
    return c1; // return this trained SVM , which is trained with this precision value C
}

void CollectPrecisionRecallData()
{
 const int N = 20;
 float precision,recall;
 Ptr<CvSVM> csvm;
 ofstream oStream("../PrecisionRecallValues.dat");
 
 float C[] = { 0.001, 0.005, 0.01, 0.05, 0.09, 0.1, 0.3, 0.5, 0.7 ,0.9, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 60.0, 80.0, 100.0 };
 
 for(int i =0; i < N ; ++i)
 {
   csvm = PredictValuesBasedOnC(C[i],precision,recall);   
   oStream << recall << " " << precision << endl;
 }
  
 oStream.close();
 cout<<"Precision-Recall Data Saved ../PrecisionRecallValues.dat"<<endl;
 
 
    
}



void ex5()
{

    cout << "Exercise 5 - Multiple Detections" << endl;
    vector< vector <float> > allResponses;

    // TODO: Write your own custom class myHogDetector which does:
    // - opencv HOGDescriptor combined with a self written sliding window
    // - multiscale detection
    // - non maximum suppression
    // (you'll need to do some more coding compared to the previous tasks)
    
    myHogDetector mHogDetector;  
    //string svmFilename = "../somePretrainedSvm.dat"; // TODO: Load some previously trained SVM
    float a,b;
    mHogDetector.setSVMDetector( PredictValuesBasedOnC(5,a,b) );

    /// Read all filenames in a folder
    string path = "../section_1_testImages/";
    string filelist = path + "filenames.txt";
    vector<string> filenames;
    readFilenamesFromAscii(filelist, filenames);
    int numImgs = (int) filenames.size();
    Mat image;
    int counter = 0;

    for (int i = 0; i < numImgs; i++) {
        string filename = path + filenames[i];
        Mat image = imread(filename);
        cout << filename << "  "<<image.rows<<","<<image.cols<<endl;

        /// TODO: Use your custom "myHogDetector mHogDetector" on the same test images as used in section 1 and display the results
	// ...
        //Rect BestDetection;
        vector<Rect> BBD;
        mHogDetector.mDetectMultiple(image,BBD);
        //mHogDetector.Test(image,BBD);
        
        
        //BBD.push_back(BestDetection);
        //drawBoundingBox("My own HOG Detector",image,BBD);
        counter++;
    }

    cout << "Done" << endl;
    waitKey(0);
    destroyAllWindows();
}




int main( int argc, char** argv)
{
    /// Section 1 - OpenCV HOG
    ex1();
    ex11();

    /// Section 2 - Extract HOG Features
    ex2();

    /// Section 3 - Train SVM
    ex3();
    CollectPrecisionRecallData();
    
    /// Section 5 - Multiple Detections
    ex5();



    return 0;
}




// Helper Functions
void readFilenamesFromAscii(string filename, vector<string>& vecFilenames)
{

	string sLine = "";
	ifstream inFile(filename.c_str());
        cout<<"\n "<<filename;
     	while (!inFile.eof())
	{
            getline(inFile, sLine);
            if (!sLine.empty())
               vecFilenames.push_back(sLine);
                   
	}
	inFile.close();

}

void saveMatAsciiWithHeader( const string& filename, Mat& matData)
{
    if (matData.empty()){
        cout<<"File could not be saved. MatData is empty"<<endl;
       return;
    }
    ofstream oStream(filename.c_str());
    // Create header
    oStream << matData.rows << " " << matData.cols << endl;
    // Write data
    for(int ridx=0; ridx < matData.rows; ridx++)
    {
        for(int cidx=0; cidx < matData.cols; cidx++)
        {
            oStream << setprecision(9) << matData.at<float>(ridx,cidx) << " ";
        }
        oStream << endl;
    }
    oStream.close();
    cout<<"Saved "<<filename.c_str()<<endl;

}

void readMatAsciiWithHeader( const string& filename, Mat& matData)
{
    cout << "Create matrix from file :"<<filename<<endl;

    ifstream iStream(filename.c_str());
    if(!iStream){
        cout<<"File cannot be found"<<endl;
        exit(-1);
    }

    int rows, cols;
    iStream >> rows;
    iStream >> cols;
    matData.create(rows,cols,CV_32F);
    cout<<"numRows: "<<rows<<"\t numCols: "<<cols<<endl;

    matData.setTo(0);
    float *dptr;
    for(int ridx=0; ridx<matData.rows; ++ridx){
        dptr = matData.ptr<float>(ridx);
        for(int cidx=0; cidx<matData.cols; ++cidx, ++dptr){
            iStream >> *dptr;
        }
    }
    cout<<"\n Read successfully....";
    iStream.close();

}
