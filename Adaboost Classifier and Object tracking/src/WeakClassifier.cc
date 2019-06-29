/*
 * NearestMeanClassifier.cc
 *
 *  Created on: Apr 25, 2014
 *      Author: richard
 */

#include "WeakClassifier.hh"
#include <cmath>
#include <iostream>
#include <stdio.h>

/*
 * Stump
 */

Stump::Stump() :
		dimension_(0),
		splitAttribute_(0),
		splitValue_(0),
		classLabelLeft_(0),
		classLabelRight_(0)
{}

void Stump::initialize(u32 dimension) {
        dimension_ = dimension;
}

f32 Stump::weightedGain(const std::vector<Example>& data, const Vector& weights, u32 splitAttribute, f32& splitValue, u32& resultingLeftLabel) {
  
    f32 gerror = 999.0;
    u32 left;
    std::vector<f32> split;
    // calculate the mean split value for this fature, as the mean of successive sorted feature value
    for( u32 i = 0; i < data.size(); ++i)
        split.push_back( data.at(i).attributes.at(splitAttribute) );
    std::sort(split.begin(), split.end());
    for( u32 i = 1; i < data.size(); ++i)
        split.at(i-1) = ( split.at(i-1) + split.at(i) ) /2.0;
       
    // for each split value test the weighted error it gives 
    for( u32 i = 0; i < data.size()-1; ++i)
    {
       f32 error ;    
       f32 error_left = 0.0;
       f32 error_right = 0.0;
       for ( u32 index = 0; index < data.size(); ++index )
       {
 
        if( data.at(index).attributes.at(splitAttribute ) < split.at(i) )
        {
          error_left +=   weights[index] * ( data.at(index).label != 0 ? 1.0 : 0.0);
          error_right +=  weights[index] * ( data.at(index).label != 1 ? 1.0 : 0.0);
        }
        else
        {
          error_left +=   weights[index] * ( data.at(index).label != 1 ? 1.0 : 0.0);
          error_right +=  weights[index] * ( data.at(index).label != 0 ? 1.0 : 0.0);
        }
       }
       //std::cout<<"\nStart:::\n"<<splitAttribute<<"   ::::"<<error_left<<" "<<error_right<<" <> "<< error_left+error_right;
       // getchar();
     
       if( error_left < error_right )
       {
         left = 0;
         error =  error_left;
       }   
       else
       {    
        left = 1;
        error =  error_right;
       }
    
       if( error < gerror)
       {
        gerror = error;
        resultingLeftLabel = left;
        splitValue = split.at(i);
       }
    
    }// 
    return gerror;
}


void Stump::train(const std::vector<Example>& data, const Vector& weights) {
    f32 global_feature_min_error=9999.0; 
    
    for ( u32 featuredim = 0; featuredim < dimension_; ++ featuredim)
    {
        u32 leftlabel,rightlabel;
        f32 splitvalue;
        f32 featureError = weightedGain(data,weights,featuredim,splitvalue,leftlabel);
        rightlabel = 1-leftlabel;
        if( featureError < global_feature_min_error)
        {
          splitAttribute_ = featuredim;
          splitValue_ = splitvalue;
          classLabelLeft_ = leftlabel;
          classLabelRight_ = rightlabel;
          global_feature_min_error = featureError;
        }
        //std::cout<<"\n values = "<<featureError<<"    "<<featuredim<<"   "<<splitvalue<<"  "<<leftlabel<<rightlabel;
    }

    //std::cout<<"\n StumpError = "<<global_feature_min_error<<"    "<<splitAttribute_<<"   "<<splitValue_<<"  "<<classLabelLeft_<<classLabelRight_;
    //getchar();
}




u32 Stump::classify(const Vector& v) {
   return ( v.at(splitAttribute_) < splitValue_ ? classLabelLeft_:classLabelRight_ );
}

void Stump::classify(const std::vector<Example>& data, std::vector<u32>& classAssignments) {
     for ( int dataInstance = 0; dataInstance < data.size(); ++dataInstance )
     {
         Vector v = data.at(dataInstance).attributes;
         classAssignments.at(dataInstance) = this->classify(v);
     }
}
