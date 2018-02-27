#ifndef _LIB_CAFFE_H_
#define _LIB_CAFFE_H_

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <memory>

class FacialCaffe
{
public:
    FacialCaffe();
    ~FacialCaffe();
    bool LoadModel(char **argv);
    std::vector<float> Predict(const cv::Mat& Image);
    std::string GetResult(std::vector<std::string> Result);
private:
    class CaffeClass;
    std::auto_ptr<CaffeClass> caffepClass;
};

#endif // _LIB_CAFFE_H_
