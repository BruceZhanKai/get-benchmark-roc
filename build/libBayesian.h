#ifndef _LIB_BAYESIAN_H_
#define _LIB_BAYESIAN_H_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <memory>

typedef std::pair<float, std::string> PAIR;

class FacialBayesian
{
public:
    FacialBayesian();
    ~FacialBayesian();
    bool LoadModel(char** argv);
    bool SaveFeature(std::vector<float> Feature, std::string Label);
    bool LoadFeature();
    std::vector<PAIR> Verify(std::vector<float> Feature);
    float Verify(std::vector<float> Feature1, std::vector<float> Feature2);

private:
    class BayesianClass;
    std::auto_ptr<BayesianClass> bayesianpClass;
};


#endif // _LIB_BAYESIAN_H_
