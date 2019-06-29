#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

/** classification header **/
#define NUM_ITERATIONS 5
#define STEP_SIZE 1

// This function is not used, it was used to test performance of opencv in matrix product 
bool DotProduct(Mat& A,Mat& B,Mat& C)
{
 //cout<<"\n Dotproduct = "<<A.rows<<"  "<<A.cols<<"  "<<B.rows<<"  "<<B.cols<<"   "<<C.rows<<"  "<<C.cols;
    if( A.cols != B.rows )
     return false;
 if( A.rows != C.rows || B.cols !=C.cols )
     return false;

 for (int i = 0; i < A.rows; i++)
 {
  for (int j = 0; j < B.cols; j++)
  {
    C.at<double>(i,j) = 0.0;
    for (int k = 0; k < A.cols; k++)
    {
      C.at<double>(i,j) += A.at<double>(i,k) * B.at<double>(k,j);
    }
  }
 }   
 return true;   
}
//////////////////////////////////////////////////////////////////////////////////////////





struct ClassificationParam{
    string posTrain, negTrain;
    string posTest, negTest;
};

// regression class for various regression methods
class LogisticRegression{
private:
    Mat train, test;                // each column is a feature vector
    Mat gtLabelTrain, gtLabelTest;  // row vector
    Mat phi;

    int loadFeatures(std::string& trainFile, std::string& testFile, Mat& feat, Mat& gtLabel);

public:
    LogisticRegression(ClassificationParam& param);
    int learnClassifier(); // TODO implement
    int testClassifier(); // TODO implement
    float sigmoid(float a);
    ~LogisticRegression(){}
};




/** regression header **/
#define FIN_RBF_NUM_CLUST 300
#define RBF_SIGMA 1e-3

// reading input parameters
struct RegressionParam{
    std::string regressionTrain;
    std::string regressionTest;
};

// models for regression
class Model{
public:
    Mat phi;        // each row models wi
    Mat sigma_sq;   // column vector
    Mat codeBook;   // codebook for finite kernel reg.
};

// regression class for various regression methods
class Regression{
private:
    Mat   trainx, trainw;
    Mat   testx, testw;
    Model linear_reg, fin_rbf_reg, dual_reg;

    int loadFeatures(std::string& fileName, Mat& vecx, Mat& vecw);

public:
    Regression(RegressionParam& param);
    ~Regression(){}
    int GenerateRBFKernal(Mat&);
    int MLE(Mat& ,Mat& ,Mat ,Mat );
    bool radial_basis_function(Mat, Mat&);
    int trainLinearRegression(); // TODO implement
    int trainFiniteRBF_KernelRegression(); // TODO implement
    int trainDualRegression(); // TODO implement

    int testLinearRegresssion(); // TODO implement
    int testFinite_RBF_KernelRegression(); // TODO implement
    int testDualRegression(); // TODO implement

};


 
int main()
{
    
    ClassificationParam cparam;
    cparam.posTrain = "../data/bottle_train.txt";
    cparam.negTrain = "../data/horse_train.txt";
    cparam.posTest  = "../data/bottle_test.txt";
    cparam.negTest  = "../data/horse_test.txt";

    LogisticRegression cls(cparam);
    cls.learnClassifier();
    cls.testClassifier();
    
    RegressionParam rparam;
    rparam.regressionTrain = "../data/regression_train.txt";
    rparam.regressionTest  = "../data/regression_test.txt";

    Regression reg(rparam);

    // linear regression
   
   
   
     reg.trainLinearRegression();
    reg.testLinearRegresssion();
    reg.trainFiniteRBF_KernelRegression();
    reg.testFinite_RBF_KernelRegression();
     reg.trainDualRegression();
    reg.testDualRegression();

    return 0;
}

/** classification functions **/
LogisticRegression::LogisticRegression(ClassificationParam& param){

    loadFeatures(param.posTrain,param.negTrain,train,gtLabelTrain);
    loadFeatures(param.posTest,param.negTest,test,gtLabelTest);
}

int LogisticRegression::loadFeatures(string& trainPos, string& trainNeg, Mat& feat, Mat& gtL){

    ifstream iPos(trainPos.c_str());
    if(!iPos) {
        cout<<"error reading train file: "<<trainPos<<endl;
        exit(-1);
    }
    ifstream iNeg(trainNeg.c_str());
    if(!iNeg) {
        cout<<"error reading test file: "<<trainNeg<<endl;
        exit(-1);
    }

    int rPos, rNeg, cPos, cNeg;
    iPos >> rPos;
    iPos >> cPos;
    iNeg >> rNeg;
    iNeg  >> cNeg;

    if(cPos != cNeg){
        cout<<"Number of features in pos and neg classes unequal"<<endl;
        exit(-1);
    }
    feat.create(cPos+1,rPos+rNeg,CV_32F); // each column is a feat vect
    gtL.create(1,rPos+rNeg,CV_32F);       // row vector


    // load positive examples
    for(int idr=0; idr<rPos; ++idr){
        gtL.at<float>(0,idr) = 1;
        feat.at<float>(0,idr) = 1;
        for(int idc=0; idc<cPos; ++idc){
            iPos >> feat.at<float>(idc+1,idr);
        }
    }

    // load negative examples
    for(int idr=0; idr<rNeg; ++idr){
        gtL.at<float>(0,rPos+idr) = 0;
        feat.at<float>(0,rPos+idr) = 1;
        for(int idc=0; idc<cNeg; ++idc){
            iNeg >> feat.at<float>(idc+1,rPos+idr);
        }
    }

    iPos.close();
    iNeg.close();

    return 0;
}

float LogisticRegression::sigmoid(float a){
    return 1.0f/(1+exp(-a));
}


int LogisticRegression::learnClassifier()
{
    cout<<"\n Training the Logistic Regression Classifier begins :::: <Set to 5 Iterations > ";
    double Alpha = 0.01;
    phi.create(train.rows,1,CV_32F);
    phi.setTo(0);
    int NoOfEpochs = NUM_ITERATIONS;
    gtLabelTrain = gtLabelTrain.t();
    for( int i = 1; i <= NoOfEpochs; i++) // Iterations for newton raphson method
    {
     cout<<"\n Beginning Iteration Number "<<i<<" :::";
     // Calculate the hessian matrix and gradient
     Mat GradientVector; GradientVector.create(train.rows,1,CV_32F); GradientVector.setTo(0);
     Mat HessianMatrix;  HessianMatrix.create(train.rows,train.rows,CV_32F);  HessianMatrix.setTo(0);
     for( int index = 0 ; index < train.cols; ++index )
     {
         Mat netm = phi.t() * train.col(index);
         double ai = sigmoid(netm.at<double>(0));
         double wi = gtLabelTrain.at<double>(index,0);
         GradientVector += (ai - wi) * train.col(index);
         HessianMatrix += ai * ( 1.0d - ai ) * ( train.col(index) * train.col(index).t() );
     }
     
     GradientVector = -1.0 * GradientVector;
     HessianMatrix = -1.0 * HessianMatrix;
     phi += Alpha * HessianMatrix.inv(DECOMP_SVD) * GradientVector;
    }
    cout<<"\n Training Over...........\n";
    return 0;
    
}

int LogisticRegression::testClassifier()
{
    int tp = 0,fp = 0, tn = 0, fn = 0;
    
    Mat netm = phi.t() * test;
    for(int i = 0; i < netm.cols; ++i)
    {
       int prediction = sigmoid(netm.at<double>(0,i))>0 ? 1 : 0 ;
       int actual     = this->gtLabelTest.at<float>(0,i);
       if( actual == 1 )
       {
           if( prediction == 1)
               tp ++;
           else
               fn ++;
       }
       else
       {
           if( prediction == 1)
               fp ++;
           else
               tn ++;
       }
           
    }
    cout<<"\n ~~~~~~Test Results~~~~~~~~~~~";
    cout<<"\n True Positives  :"<<tp;
    cout<<"\n False Positives :"<<fp;
    cout<<"\n True Negatives  :"<<tn;
    cout<<"\n False Negatives :"<<fn;
    float accuracy = (float)(tp+tn) / (float)(tp+fp+tn+fn);
    cout<<"\n Accuracy = "<<accuracy<<"\n ";
    if(accuracy >=0.5)
        cout<<"\n It is a good classifictaion.";
    else
        cout<<"\n It is a bad classification.";
    return 0;
}


/** regression functions **/
Regression::Regression(RegressionParam& param){
    // load features
    loadFeatures(param.regressionTrain,trainx,trainw);
    loadFeatures(param.regressionTest,testx,testw);
//    cout<<"features loaded successfully"<<endl;

    // model memory
    linear_reg.phi.create(trainx.rows,trainw.rows,CV_32F); linear_reg.phi.setTo(0);
    linear_reg.sigma_sq.create(trainw.rows,trainw.rows,CV_32F); linear_reg.sigma_sq.setTo(0);
    fin_rbf_reg.phi.create(FIN_RBF_NUM_CLUST,trainw.rows,CV_32F);
    fin_rbf_reg.sigma_sq.create(trainw.rows,trainw.rows,CV_32F);
    dual_reg.phi.create(trainx.cols,trainw.rows,CV_32F);dual_reg.phi.setTo(0);
    dual_reg.sigma_sq.create(trainw.rows,trainw.rows,CV_32F); dual_reg.sigma_sq.setTo(0);

}
int Regression::loadFeatures(string& fileName, Mat& matx, Mat& matw){

    // init dimensions and file
    int numR, numC, dimW;
    ifstream iStream(fileName.c_str());
    if(!iStream){
        cout<<"cannot read feature file: "<<fileName<<endl;
        exit(-1);
    }
    
    // read file contents
    iStream >> numR;
    iStream >> numC;
    iStream >> dimW;
    matx.create(numC-dimW+1,numR,CV_32F); // each column is a feature
    matw.create(dimW,numR,CV_32F);        // each column is a vector to be regressed

    for(int r=0; r<numR; ++r){
        // read world data
        for(int c=0; c<dimW; ++c)
            iStream >> matw.at<float>(c,r);
        // read feature data
        matx.at<float>(0,r)=1;
        for(int c=0; c<numC-dimW; ++c)
            iStream >> matx.at<float>(c+1,r);
    }
    iStream.close();

    return 0;
}

// Function to perform Maximum Likelihood Estimation in linear regression
int Regression::MLE(Mat& _phi,Mat& _sigma_square,Mat X,Mat w)
{  
  
   //Calculate phi
   _phi = ((X*X.t()).inv(DECOMP_SVD)*X)*w.t();
  
   //Calculate sigma _sigma_square
   int N = X.cols;
   Mat p = w.t() - X.t() * _phi;
   _sigma_square = (p.t() * p)/N; 
   return 0;
   
}

int Regression::trainLinearRegression()
{
    
   MLE(this->linear_reg.phi,this->linear_reg.sigma_sq,this->trainx,this->trainw);
   return 0; 
}

int Regression::testLinearRegresssion()
{
    
   Mat w_ = (this->testx.t())*(this->linear_reg.phi);
 
   //Finding the mean squared error
   Mat error = (w_ - testw.t());
   //cout<<"\n Error Matrix of Linear Regression =\n"<<error;
   
   double Mse = 0.0;
   for (int i = 0; i < error.rows; i++)
   {
    for (int j = 0; j < error.cols; j++)
    {
        Mse += (error.at<double>(i,j));
    }
   }
   Mse /= ( error.rows * error.cols );
   cout<<"\n Mean Squared Error of Linear Regression = "<<Mse;
   return 0;
}


bool Regression::radial_basis_function(Mat x,Mat&z)
{
    Mat CLusterlabels;
    Mat RBFcenters;
    
    //Use opencv k-means clustering to learn the RBF centres and get the CLusterlabels 
    cv::kmeans( x.t(), 
                FIN_RBF_NUM_CLUST, 
                CLusterlabels, 
                TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
                10,
                cv::KMEANS_RANDOM_CENTERS, 
                RBFcenters
              );
    
    z.create(x.rows, x.cols, CV_32F);
    z.setTo(0);
   
    for(int i = 0; i < z.rows; i++)
    {
        for(int j = 0; j < z.cols; j++)
        {
          double sum = 0.0;
          for(int index = 0; index < z.rows; index++)
             sum += pow( (x.at<float>(i, j) - RBFcenters.at<float>(CLusterlabels.at<int>(j, 0))) , 2.0 );
          z.at<float>(i, j) = exp(-(sum)/RBF_SIGMA);
        }
    }
    //z contains the transformed function, using RBF kernal
    return true;
}


int Regression::trainFiniteRBF_KernelRegression()
{
    Mat z ;
    radial_basis_function(this->trainx,z); // Transform x into z , using radial basis kernal 
    MLE(this->fin_rbf_reg.phi,this->fin_rbf_reg.sigma_sq,z,this->trainw);
    return 0;
}

int Regression::testFinite_RBF_KernelRegression()
{
    Mat w_ = (this->testx.t())*(this->fin_rbf_reg.phi);
 
   //Finding the mean squared error
   Mat error = (w_ - testw.t());
   //cout<<"\nError Matrix of Non-linear Regression RBF = "<<error;
   
   double Mse = 0.0;
   for (int i = 0; i < error.rows; i++)
   {
    for (int j = 0; j < error.cols; j++)
    {
        Mse += (error.at<double>(i,j));
    }
   }
   Mse /= ( error.rows * error.cols );
    
   cout<<"\n Mean Squared Error of Non-Linear Regression ( using RBF kernal )  = "<<Mse;
   return 0;
}

int Regression::GenerateRBFKernal(Mat& k)
{
  
  for (int i = 0; i < trainx.cols; i++)
  {
    for (int j = 0; j < trainx.cols; j++)
    {
       Mat temp;
       temp =  (trainx.col(i) - trainx.col(j));
       temp = temp.t() * temp;
       float p =  (-0.5 /( RBF_SIGMA * RBF_SIGMA ) ) * temp.at<float>(0);
       k.at<float>(i,j) =  exp(p);
    }
  }
  return 0;
}

int Regression::trainDualRegression()
{
   //Calculate phi
   Mat Z ;
   Z.create(trainx.cols,trainx.cols,CV_32F);
   // Use a radial basis kernal 
   GenerateRBFKernal(Z);
   Mat w = trainw;
   Mat DualParam = Z.inv(DECOMP_SVD) * w.t();
   Mat T = w.t() - Z * DualParam;
   //Calculate sigma
   if ( this->trainx.cols < this->trainx.rows )
      this->dual_reg.sigma_sq =  ( T.t() * T ) /  (this->trainx.cols);
   else // since D > I, sigma_sq = 0 in this case
   {
      this->dual_reg.sigma_sq.create(trainw.rows,trainw.rows,CV_32F);
      this->dual_reg.sigma_sq.setTo(0);
   }
   this->dual_reg.phi  = (this->trainx)* DualParam;
   return 0;
}

int Regression::testDualRegression()
{
    
  Mat w_ = (this->testx.t())*(this->dual_reg.phi);
  
   //Finding the mean squared error
   Mat error = (w_ - testw.t());
   //cout<<"\nError Matrix of Dual Regression ="<<error;
   
   double Mse = 0.0;
   for (int i = 0; i < error.rows; i++)
   {
    for (int j = 0; j < error.cols; j++)
    {
        Mse += pow(error.at<double>(i,j),2.0);
    }
   }
   
   Mse /= ( error.rows * error.cols );
   Mse = pow(Mse,0.5);
    
   cout<<"\n Mean Squared Error of Dual Regression (using RBF kernal ) = "<<Mse;
   return 0;   
}

