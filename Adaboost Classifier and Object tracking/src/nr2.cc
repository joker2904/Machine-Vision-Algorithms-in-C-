#include <iostream>
#include <string.h>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include "Types.hh"
#include "AdaBoost.hh"

void readData(const char* filename, std::vector<Example>& data) {
	std::ifstream f(filename);
	u32 dimension, nObservations;
	f >> dimension;
	f >> nObservations;
	data.resize(nObservations);
	for (u32 i = 0; i < nObservations; i++) {
		f >> data.at(i).label;
		data.at(i).attributes.resize(dimension);
		for (u32 d = 0; d < dimension; d++) {
			f >> data.at(i).attributes.at(d);
		}
	}
	f.close();
}

int main(int argc, const char* argv[]) {
	// implement the functions in AdaBoost.cc (class containing several weak classifiers) and WeakClassifier.cc to make this work
	// or implement your own solution from scratch
	if (argc != 4) {
		std::cout << "usage: " << argv[0] << " <train-set> <test-set> <# iterations>" << std::endl;
		exit(1);
	}
	const char* trainFile = argv[1];
	const char* testFile = argv[2];
	u32 adaBoostIterations = atoi(argv[3]);

	std::vector<Example> trainingData;
	std::vector<Example> testData;
	readData(trainFile, trainingData);
	readData(testFile, testData);

	// train cascade of weak classifiers
	AdaBoost adaBoost(adaBoostIterations,0);
	adaBoost.initialize(trainingData);
	adaBoost.trainCascade(trainingData);

	// classification on test data
	u32 nClassificationErrors = 0;
	for (u32 i = 0; i < testData.size(); i++) {
		u32 c = adaBoost.classify(testData.at(i).attributes);
		nClassificationErrors += (c == testData.at(i).label ? 0 : 1);
	}
	f32 accuracy = 1.0 - (f32) nClassificationErrors / (f32) testData.size();

	std::cout << "Classified " << testData.size() << " examples." << std::endl;
	std::cout << "Accuracy: " << accuracy << " (" << testData.size() - nClassificationErrors << "/" << testData.size() << ")" << std::endl;
        
        // analyzing k and the ClassificationErrors
        // train cascade of weak classifiers
        int k[] = {1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50};
        std::cout<<"k :  Accuracy"<<std::endl;
        std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
        for ( int i = 0; i < 11; ++i)
        {
          adaBoostIterations = k[i];
	  AdaBoost adaBoostTester(adaBoostIterations,0);
	  adaBoostTester.initialize(trainingData);
	  adaBoostTester.trainCascade(trainingData);

	  // classification on test data
	  u32 nClassificationErrors = 0;
	  for (u32 i = 0; i < testData.size(); i++) {
		u32 c = adaBoostTester.classify(testData.at(i).attributes);
		nClassificationErrors += (c == testData.at(i).label ? 0 : 1);
	  }
	  f32 accuracyfork = 1.0 - (f32) nClassificationErrors / (f32) testData.size();
          std::cout <<adaBoostIterations<< " : "<<accuracyfork << " (" << testData.size() - nClassificationErrors << "/" << testData.size() << ")"<<std::endl ;
        }        
	return 0;
}
