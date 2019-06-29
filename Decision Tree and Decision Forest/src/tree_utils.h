#ifndef TREE_UTILS_H
#define TREE_UTILS_H
#include <Sample.h>
#include <set>
#include <map>


// labels for classes
enum {VOID = 0, SHEEP = 1, WATER = 2, GRASS = 3};

/// structure to contain parameters for the tree 
struct TreeParam{
    int depthOfTree = 15;
    int minImagePatchesAtLeaf = 20;
    int ImagePatchDimensions = 16;
    int numOfClasses = 4;
    int minTrainPatchesPerClass = 100;
};

#endif // TREE_UTILS_H





