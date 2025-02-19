/*
 * AdaBoost.hh
 *
 *  Created on: Apr 25, 2014
 *      Author: richard
 */

#ifndef ADABOOST_HH_
#define ADABOOST_HH_

#include <vector>
#include <string.h>
#include <math.h>
#include "Types.hh"
#include "WeakClassifier.hh"

class AdaBoost
{
private:
	u32 nIterations_;
        u32 nk_;
	Vector weights_;
	std::vector<Stump> weakClassifier_;
	Vector classifierWeights_;

	void normalizeWeights(f32);
	void updateWeights(const std::vector<Example>& data, const std::vector<u32>& classAssignments, u32 iteration);
	f32 weightedErrorRate(const std::vector<Example>& data, const std::vector<u32>& classAssignments);
public:
	AdaBoost(u32 nIterations,u32 k);
	void initialize(std::vector<Example>& data);
	void trainCascade(std::vector<Example>& data);
	u32 classify(const Vector& v);
	f32 confidence(const Vector& v, u32 k);
        void retrain(std::vector<Example>& data);
};


#endif /* ADABOOST_HH_ */
