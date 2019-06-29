#include "HogDetector.h" 


/*
 1. Take an scaled image.
 2. Scan all the windows of the image
 3. For each window the the hog descriptor.
 4. Use the trained SVM to predict the label, and get the Cost/
 5. Use ratios to estimate distance of this rectangle for the top-left corner of the image. 
 * */
/*
bool customSort( std::pair< float,std::pair<int,int> > p1, 
                 std::pair< float,std::pair<int,int> > p2)
{
  if( (p1.first) <  (p2.first) )
      return true;
  else
      return false;
}

*/
void drawbestBoundingBox(string DisplayName,Mat& image,std::map< std::pair<int,int> , Rect >& found_filtered,int ScaleFactor)
{
  static int COUNTER = 0;
 // cv::namedWindow( DisplayName, cv::WINDOW_AUTOSIZE);
  //std::sleep(1);
  for( std::map< std::pair<int,int> , Rect >::iterator itr = found_filtered.begin(); itr != found_filtered.end(); ++itr)
  {
     cv::rectangle(image,itr->second, 0, 3);     
  }
  //cv::imshow(DisplayName, image);
  ++COUNTER;
  string S = "../Detections/"+DisplayName+to_string(COUNTER)+".bmp";
  cv::imwrite(  cv::format( S.c_str() ) , image );
  //cv::waitKey(0);   
}

/*
float findIntersection(Rect& A,Rect& B)
{
  if( ( B.x <= A.x && A.x <= B.x+B.width ) &&   ( B.y <= A.y && A.y <= B.y+B.height ) )
      return (B.width - ( A.x-B.x)) * (B.height - ( A.y-B.y)) ; 
  
  if( ( B.x <= A.x && A.x <= B.x+B.width ) &&   ( B.y <= A.y && A.y <= B.y+B.height ) )
      return (B.width - ( A.x-B.x)) * (B.height - ( A.y-B.y)) ;   
}
*/

void  myHogDetector::EliminateOverlappingRectangles( int row,int col, float** LabelMatrix,
                                     float** PredictionMatrix,
                                     std::map< float, std::pair<int,int> >& PredictionLabelMap,
                                     std::map< std::pair<int,int> , Rect >& DetectionMap
                                   )
{
    /* Remove by area overlapping */
  Rect S = DetectionMap[ std::pair<int,int>(row,col) ] ;
  for( std::map< std::pair<int,int> , Rect >::iterator itr = DetectionMap.begin(); itr != DetectionMap.end(); ++itr)
  {
    int rowL = (itr->first).first;
    int colL = (itr->first).second;
    // row and col is the index of the highest prediction till date 
    if( row == rowL && col == colL )
        continue;
    Rect R = DetectionMap[ std::pair<int,int>(rowL,colL) ];
    
    Rect I = S & R;
    Rect U = S | R;
     
      float p = (float)((float)I.area() / (float)U.area());
    
    if( p >= 0.005)
       DetectionMap.erase( std::pair<int,int>(rowL,colL) );
    
  }
    
    
    
    
    /* Remove by prediction score */
    if ( row >0 && col >0 && LabelMatrix[row-1][col-1] == 1 )
        if ( PredictionMatrix[row-1][col-1] < PredictionMatrix[row][col] )
        {
            DetectionMap.erase( std::pair<int,int>(row-1,col-1) );
            PredictionLabelMap.erase( PredictionMatrix[row-1][col-1] );
        }
    if ( col >0 && LabelMatrix[row][col-1] == 1 )
        if ( PredictionMatrix[row][col-1] < PredictionMatrix[row][col] )
        {
            DetectionMap.erase( std::pair<int,int>(row,col-1) );
            PredictionLabelMap.erase( PredictionMatrix[row][col-1] );
        }
    if ( row >0 && LabelMatrix[row-1][col] == 1 )
        if ( PredictionMatrix[row-1][col] < PredictionMatrix[row][col] )
        {
            DetectionMap.erase( std::pair<int,int>(row-1,col) );
            PredictionLabelMap.erase( PredictionMatrix[row-1][col] );
        }
     if ( LabelMatrix[row+1][col] == 1 )
        if ( PredictionMatrix[row+1][col] < PredictionMatrix[row][col] )
        {
            DetectionMap.erase( std::pair<int,int>(row+1,col) );
            PredictionLabelMap.erase( PredictionMatrix[row+1][col] );
        }
    if (  col >0 && LabelMatrix[row+1][col-1] == 1 )
        if ( PredictionMatrix[row+1][col-1] < PredictionMatrix[row][col] )
        {
            DetectionMap.erase( std::pair<int,int>(row+1,col-1) );
            PredictionLabelMap.erase( PredictionMatrix[row+1][col-1] );
        }
    if ( LabelMatrix[row][col+1] == 1 )
        if ( PredictionMatrix[row][col+1] < PredictionMatrix[row][col] )
        {
            DetectionMap.erase( std::pair<int,int>(row,col+1) );
            PredictionLabelMap.erase( PredictionMatrix[row][col+1] );
        }
    if ( LabelMatrix[row+1][col+1] == 1 )
        if ( PredictionMatrix[row+1][col+1] < PredictionMatrix[row][col] )
        {
            DetectionMap.erase( std::pair<int,int>(row+1,col+1) );
            PredictionLabelMap.erase( PredictionMatrix[row+1][col+1] );
        }
     if ( row >0  && LabelMatrix[row-1][col+1] == 1 )
        if ( PredictionMatrix[row-1][col+1] < PredictionMatrix[row][col] )
        {
            DetectionMap.erase( std::pair<int,int>(row-1,col+1) );
            PredictionLabelMap.erase( PredictionMatrix[row-1][col+1] );
        }
}

float** myHogDetector::Allocate(int m,int n)
{
  float** R = (float**)malloc( sizeof(float*) * m);
  for( int i = 0; i < m ; ++i)
      R[i] = (float*)malloc( sizeof(float) * n);
  
  return R;  
}

bool myHogDetector::Prediction(Mat& image,float& Cost,float& label)
{
    HOGDescriptor hog;
    
    vector<float> descriptors;
    hog.compute(image, descriptors,Size(5,5),Size(0,0));
    //cout<<"\n Window rows ="<<image.rows<<" , Window cols="<<image.cols<<"\ndescriptor="<<descriptors.size()<<"....end";
    Mat temp = Mat(1, lengthHogDescriptor, CV_32F, &descriptors.front()) ;
    //cout<<"\nMat ::"<<temp;
    if( TrainedSVM != NULL )
    {
      label = TrainedSVM->predict(temp,false);
      Cost = TrainedSVM->predict(temp,true);
    }
    else
    cout<<"\n Null pointer exception....svm pointer is null ";
}



float myHogDetector::MaximumSuppressionArea( float** LabelMatrix,
                                             float** PredictionMatrix,
                                             std::map< float, std::pair<int,int> >& PredictionLabelMap,
                                             std::map< std::pair<int,int> , Rect >& DetectionMap,
                                             std::vector<Rect>& BestDetections
                                           )
{
  // std::sort( PredictionLabelMap.begin(), PredictionLabelMap.end() , customSort);
  for( std::map< float, std::pair<int,int> >::reverse_iterator itr = PredictionLabelMap.rbegin(); itr != PredictionLabelMap.rend(); ++itr)
  {
    //   cout<<endl<<itr->first<<" :: "<<PredictionLabelMap[itr->first].first<<" , "<<PredictionLabelMap[itr->first].second;
    int row = PredictionLabelMap[itr->first].first;
    int col = PredictionLabelMap[itr->first].second;
    // row and col is the index of the highest prediction till date 
    EliminateOverlappingRectangles(row,col,LabelMatrix,PredictionMatrix,PredictionLabelMap,DetectionMap);
    
  }
    
}


bool myHogDetector::ScanScaledImage( Mat& image, 
                                     float** LabelMatrix,
                                     float** PredictionMatrix,
                                     std::map< float, std::pair<int,int> >& PredictionLabelMap,
                                     std::map< std::pair<int,int> , Rect >& DetectionMap,
                                     std::map< std::pair<int,int>, float >& PredictionMap
                                   )
{
//  cout<<"\n Scale Size ="<<image.rows<<" , "<<image.cols;  
  int height = 128; 
  int width = 64; 
  int row=0,col=0;
  for( int i = 0; i < (image.rows - height); i += 64)
  {
  //  cout<<endl;  
    for( int j = 0; j < (image.cols - width); j += 16)
    {
      Mat ROI = image( cv::Rect(j, i , width,height) & cv::Rect(0, 0, image.cols, image.rows)  );
      float cost,label;
      Prediction( ROI, cost,label );
      //cout<< (label==1 ? 1:0); 
      if( label == 1.0 )
      {
        //cost *= -1.0;
        PredictionLabelMap.insert( std::pair < float, std::pair<int,int>   > ( cost, std::pair<int,int>(j,i)  ) );    
        PredictionMap.insert( std::pair < std::pair<int,int> , float  > ( std::pair<int,int>(j,i)  ,   cost                      ) );  
        DetectionMap.insert ( std::pair < std::pair<int,int> , Rect   > ( std::pair<int,int>(j,i)  ,   Rect(j, i , width,height) ) ); 
        PredictionMatrix [row][col] = cost;
      }
      else
      {
         PredictionMatrix [row][col] = 0.0;
         //PredictionLabelMap.insert( std::pair < float, std::pair<int,int>   > ( cost, std::pair<int,int>(row,col)  ) ); 
      }
      LabelMatrix[row][col] = label;
      col++;         
    }
    row++;
    col=0;
  }
  
}

void  myHogDetector::EliminateOverlappingRectangles( std::map< float, std::pair<int,int> >& PredictionLabelMap,
                                                     std::map< std::pair<int,int> , Rect >& DetectionMap,
                                                     std::map< std::pair<int,int> , float>& PredictionMap,
                                                     std::vector<Rect>& Detections,
                                                     Mat& image
                                                   )
{
  cout<<"\n Elimiminate overlapping rectangles using Non Maxima Suppression ---";
  //for( std::map< float, std::pair<int,int> >::reverse_iterator itr1 = PredictionLabelMap.rbegin(); itr1 != PredictionLabelMap.rend(); ++itr1)
      
  for( std::map< std::pair<int,int> , Rect >::iterator itr1 = DetectionMap.begin(); itr1 != DetectionMap.end(); ++itr1)    
  {
   // cout<<endl<<itr1->first<<" :: "<<PredictionLabelMap[itr1->first].first<<" , "<<PredictionLabelMap[itr1->first].second;
   // int row = PredictionLabelMap[itr1->first].first;
   // int col = PredictionLabelMap[itr1->first].second;
    // row and col is the index of the highest prediction till date 
    /* Remove by area overlapping */
       
   
  //  for( std::map< std::pair<int,int> , Rect >::iterator itr1 = DetectionMap.begin(); itr1 != DetectionMap.end(); ++itr1)
   // {
    int row = (itr1->first).first;
    int col = (itr1->first).second;
    Rect S = DetectionMap[ std::pair<int,int>(row,col) ] ;  
    Mat ROI = image( S & cv::Rect(0, 0, image.cols, image.rows)  );
    if( S.width == 0  || S.height == 0  )
           continue;
    float cost,label;
    Prediction( ROI, cost,label );
   // float cost = PredictionMap[ std::pair<int,int>(row,col) ];
    for( std::map< std::pair<int,int> , Rect >::iterator itr = DetectionMap.begin(); itr != DetectionMap.end(); ++itr)
    {  
      int rowL = (itr->first).first;
      int colL = (itr->first).second;
     // float costL = PredictionMap[ std::pair<int,int>(rowL,colL) ];
      if( row == rowL && col == colL )
        continue;
      Rect R = DetectionMap[ std::pair<int,int>(rowL,colL) ];
    
      Rect I = S & R;
      Rect U = S | R;
     
      Mat ROIL = image( R & cv::Rect(0, 0, image.cols, image.rows)  );
      // cout<<"\n"<<rowL<<" ,"<<colL<<" ---------( "<<R.x<<" , "<<R.y<<") .....( "<<R.width<<" , "<<R.height <<" )";
      float costL,labelL;
      if( R.width == 0  || R.height == 0  )
           continue;
      Prediction( ROIL, costL,labelL );
      // cout<<"\n -> "<<label<<" , "<<cost; 
      // if ( labelL == -1 && cost > 0.05)
      
      
      
      float p = (float)((float)I.area() / (float)U.area());
     // cout <<endl<< I.area()<<"   "<<U.area()<<"    "<< p;
      if( p >= 0.0005) // Area of OverLap 
      {
          
          
       if ( label == 1 && labelL == 1 )
       { 
         if( cost > costL)
         {
          DetectionMap.erase( std::pair<int,int>(rowL,colL) );
          PredictionMap.erase( itr->first );
         }
         else
         {
          DetectionMap.erase( std::pair<int,int>(row,col) );
          PredictionMap.erase( itr1->first );
          break;
         }
       }   
       
       if( label == -1 && labelL == -1 )
       { 
         if( cost < costL)
         {
          DetectionMap.erase( std::pair<int,int>(rowL,colL) );
          PredictionMap.erase( itr->first );
         }
         else
         {
          DetectionMap.erase( std::pair<int,int>(row,col) );
          PredictionMap.erase( itr1->first );
          break;
         }
       } 
      
      
      if( label == 1 && labelL == -1 )
      { 
         //if( cost < costL)
         {
          DetectionMap.erase( std::pair<int,int>(rowL,colL) );
          PredictionMap.erase( itr->first );
         }/*
         else
         {
          DetectionMap.erase( std::pair<int,int>(row,col) );
          PredictionMap.erase( itr1->first );
          break;
         }*/
       }
       
      if( label == -1 && labelL == 1 )
      { /*
         if( cost < costL)
         {
          DetectionMap.erase( std::pair<int,int>(rowL,colL) );
          PredictionMap.erase( itr->first );
         }
         else*/
         {
          DetectionMap.erase( std::pair<int,int>(row,col) );
          PredictionMap.erase( itr1->first );
          break;
         }
       } 
       
      }
          
    }
  }
  //}
 
}

bool myHogDetector::mDetectMultiple(Mat& image,vector<Rect>& BestDetection)
{
   int x = image.cols;
   int y = image.rows;
   int ScaleFactor = 0;
   std::map< float, std::pair<int,int> > GlobalPredictionLabelMap;
   std::map< std::pair<int,int> , Rect > GlobalDetectionMap;
   std::map< std::pair<int,int> , float > GlobalPredictionMap;
    vector<Rect> Detections , BestDetections;
   while( x >= 64 && y >=128)
   {
     Mat scaledImage;
     resize(image, scaledImage, Size(x, y));
     //imshow("resize", scaledImage);
     //waitKey(0);
     
     int Sizex = 2000; //(x-64)/16;
     int Sizey = 2000; //(y-128)/64;
     vector<Rect> Detections , BestDetections;
     
     float** LabelMatrix = Allocate(Sizex, Sizey) ; //, CV_64FC1, cv::Scalar(0) );
     float** PredictionMatrix =  Allocate(Sizex, Sizey) ; //, CV_64FC1, cv::Scalar(0) );
     std::map< float, std::pair<int,int> > PredictionLabelMap;
     std::map< std::pair<int,int> , Rect  > DetectionMap;
     std::map< std::pair<int,int> , float > PredictionMap;
     ScanScaledImage(scaledImage, LabelMatrix, PredictionMatrix, PredictionLabelMap, DetectionMap, PredictionMap);
    // MaximumSuppressionArea(LabelMatrix, PredictionMatrix,PredictionLabelMap,DetectionMap,BestDetections);
   
     
     // This is test code. if the different scaled images and their detections want to be viewed then they next line can be commented
     // It is a very important line to debug and test the detections :) 
     
     drawbestBoundingBox("ScaledImage_",scaledImage,DetectionMap,ScaleFactor);
            
    
     // for testing purpose 
     //cout<<"\n Print the list of prediction cost ::\n";
     int p = (int)pow(1.2,ScaleFactor);
     for( std::map< float, std::pair<int,int> >::iterator itr = PredictionLabelMap.begin(); itr != PredictionLabelMap.end(); ++itr)
     {
       //   cout<<endl<<itr->first<<" :: "<<PredictionLabelMap[itr->first].first<<" , "<<PredictionLabelMap[itr->first].second;
       int row = PredictionLabelMap[itr->first].first;
       int col = PredictionLabelMap[itr->first].second;
       // cout<<"\n"<< itr->first <<" ,( "<<row<<","<<col<<")";
      
      GlobalPredictionLabelMap.insert( std::pair<float, std::pair<int,int> >(itr->first, std::pair<int,int>( (row*p),(col*p) )  ) );                                                                                                                   
                                                                                                        
                                                   
                                     
     }
     
     for( std::map< std::pair<int,int> , Rect >::iterator itr = DetectionMap.begin(); itr != DetectionMap.end(); ++itr)
     {
  
       //GlobalDetectionMap.insert(*itr);
       int rowL = (itr->first).first ;
       int colL = (itr->first).second ;
       Rect R = DetectionMap[ std::pair<int,int>(rowL,colL) ];
       Rect RS( R.x * p, R.y * p, R.width * p, R.height * p );
       GlobalDetectionMap.insert( std::pair< std::pair<int,int> , Rect>( std::pair<int,int>(rowL*p ,colL*p  ), RS  )  );
     
     }
     
     for( std::map< std::pair<int,int> , float >::iterator itr = PredictionMap.begin(); itr != PredictionMap.end(); ++itr)
     {
  
       //GlobalDetectionMap.insert(*itr);
       int rowL = (itr->first).first ;
       int colL = (itr->first).second ;
       float cost = PredictionMap[ std::pair<int,int>(rowL,colL) ];
      GlobalPredictionMap.insert( std::pair< std::pair<int,int> , float>( std::pair<int,int>(rowL*p ,colL*p  ), cost  )  );
     
     }
     
     x = (int)((float)x / (float)1.2);
     y = (int)((float)y / (float)1.2);
     ScaleFactor++;
     
   }
   
   /*
  // EliminateOverlappingRectangles(GlobalPredictionLabelMap,GlobalDetectionMap,GlobalPredictionMap,BestDetections);
   for( std::map< std::pair<int,int> , Rect >::iterator itr = GlobalDetectionMap.begin(); itr != GlobalDetectionMap.end(); ++itr)
     {
       int rowL = (itr->first).first ;
       int colL = (itr->first).second ;
       Rect R = GlobalDetectionMap[ std::pair<int,int>(rowL,colL) ];
       if( R.width == 0  || R.height == 0  )
           continue;
       Mat ROI = image( R & cv::Rect(0, 0, image.cols, image.rows)  );
       cout<<"\n"<<rowL<<" ,"<<colL<<" ---------( "<<R.x<<" , "<<R.y<<") .....( "<<R.width<<" , "<<R.height <<" )";
       float cost,label;
       Prediction( ROI, cost,label );
       cout<<"\n -> "<<label<<" , "<<cost; 
       if ( label == -1 && cost > 0.05)
       {
         GlobalDetectionMap.erase(itr);
         GlobalPredictionMap.erase(  std::pair<int,int>(rowL,colL)  );
       }
     
     }*/
   
    EliminateOverlappingRectangles(GlobalPredictionLabelMap,GlobalDetectionMap,GlobalPredictionMap,BestDetections,image);
   
   drawbestBoundingBox("Final_Detector_",image,GlobalDetectionMap,ScaleFactor);
   
  //BestDetection.push_back( p );
}


bool myHogDetector::Test(Mat& image,vector<Rect>& BestDetection)
{
   //resize(image, image, Size(64, 128));
   //cv::imshow("resize", image);
   //cv::waitKey(0);
   int height = 128; //(int)((float)image.rows / (float)1.2);
  int width = 64; //(int)((float)image.cols / (float)1.2);
 
  cout<<"\n Scaled Image size == "<<image.rows<<" , "<<image.cols;
  float CostMin = 0, label,Cost; 
  Rect p;
  for( int i = 0; i < (image.rows - height); i += 64)
  {
    cout<<endl;
    for( int j = 0; j < (image.cols - width); j += 16)
    {
       
        Mat ROI = image( cv::Rect(j, i , width,height) & cv::Rect(0, 0, image.cols, image.rows)  );
        Prediction(ROI,Cost,label);
        if ( label == 1.0 )
        {
          cout<<"\n"<<label<<" , "<<Cost<<" , "<<CostMin;
          if( -1*Cost > CostMin)    
          {
            //  Cost = CostMin;
            //  p = Rect(j, i , width,height);
            BestDetection.push_back(  Rect(j, i , width,height) );          
          }
        }
      
         
   }
  }
 // BestDetection.push_back( p );
}