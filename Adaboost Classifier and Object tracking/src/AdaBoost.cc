/*
 * AdaBoost.cc
 *
 *  Created on: Apr 25, 2014
 *      Author: richard
 */

#include "AdaBoost.hh"
#include <iostream>
#include <cmath>

AdaBoost::AdaBoost(u32 nIterations,u32 k):nIterations_(nIterations),nk_(k)
{}

void AdaBoost::normalizeWeights(f32 Z) {
    for (u32 i = 0; i < weights_.size(); i++) {
		weights_.at(i) /=  Z; 
        }
}

void AdaBoost::updateWeights(const std::vector<Example>& data, const std::vector<u32>& classAssignments, u32 iteration) {
    f32 Z = 0.0;
    f32 error_t = weightedErrorRate(data,classAssignments);
    classifierWeights_.at(iteration) = error_t / ( 1.0 - error_t); 
    for (u32 i = 0; i < weights_.size(); i++) {
                f32 v = abs( classAssignments.at(i) - data.at(i).label );
		weights_.at(i) = weights_.at(i) * pow( classifierWeights_.at(iteration) , 1.0 - v );
                Z += weights_.at(i);
	}
    normalizeWeights(Z);

}

f32 AdaBoost::weightedErrorRate(const std::vector<Example>& data, const std::vector<u32>& classAssignments) {
    f32 eta = 0.0;
    for (u32 i = 0; i < weights_.size(); i++) {	
        eta += weights_.at(i) * ( classAssignments.at(i) == data.at(i).label ? 0:1 );
	}
    return eta;
}

void AdaBoost::initialize(std::vector<Example>& data) {
	// initialize weak classifiers
        weakClassifier_.resize(nIterations_);
	for (u32 iteration = 0; iteration < nIterations_; iteration++) {
                Stump temp = Stump();
                //std::cout<<data.at(0).attributes.size()<<" ";
                temp.initialize(data.at(0).attributes.size());
		weakClassifier_.at(iteration) = temp;
                
	}
	// initialize classifier weights
	classifierWeights_.resize(nIterations_);
        // initialize weights
	weights_.resize(data.size());
	for (u32 i = 0; i < data.size(); i++) {
		weights_.at(i) = ( 1.0 / data.size() ); // Uniform distribution 
	}
	
}

void AdaBoost::trainCascade(std::vector<Example>& data) {
    for(u32 iteration = 0; iteration < nIterations_; ++iteration )
    {
        weakClassifier_.at(iteration).train(data,weights_); // train the ith weak classifer using the current arrangement of weights 
        
        std::vector<u32> classAssignments;
        classAssignments.resize( data.size() );
        
        weakClassifier_.at(iteration).classify(data,classAssignments);
        /*
        std::cout<<"\n Weights at "<<iteration<<" ::";
        for (u32 i = 0; i < data.size(); i++) {
		std::cout<<weights_.at(i)<<" ,"; 
	}*/
        updateWeights(data,classAssignments,iteration);
    }
}

void AdaBoost::retrain(std::vector<Example>& data)
{
    // discard first k weak classifiers
    
    // Retrain 
    // add them to ensemble, by re adjusting the weights 
}


u32 AdaBoost::classify(const Vector& v) {
    return confidence(v,1) > confidence(v,0) ? 1 : 0;
}

// v is a singlel instance of data . k is its label 
f32 AdaBoost::confidence(const Vector& v, u32 k) {
    f32 sum = 0.0;
    for (u32 iteration = 0; iteration < nIterations_; iteration++) {
        sum += log2( 1 / classifierWeights_.at(iteration) ) * ( weakClassifier_.at(iteration).classify(v) == k ? 1:0 );
	}
    return sum;
}
