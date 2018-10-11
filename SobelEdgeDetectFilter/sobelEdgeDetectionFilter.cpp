//
// C++ implementation of sobel edge detection filter
//
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>

int MAX_KERNEL_LENGTH = 9;     

using namespace std;
using namespace cv;
struct timeval t1, t2;

// The wrapper is used to call sharpening filter 
extern "C" void sobelFilter_CPU(const cv::Mat& input, cv::Mat& output)
{
   Point anchor = Point( -1, -1 );
   double delta = 0;
   int ddepth = -1;
   int kernel_size;

   int64 t0 = cv::getTickCount();

   /// Update kernel size for a normalized box filter
   kernel_size = 3; 
   
   cv::Mat output1;
   cv::Mat kernel1 = (Mat_<double>(kernel_size,kernel_size) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
   /// Apply 2D filter
   filter2D(input, output1, ddepth, kernel1, anchor, delta, BORDER_DEFAULT );

  
   cv::Mat output2;
   cv::Mat kernel2 = (Mat_<double>(kernel_size,kernel_size) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
   /// Apply 2D filter
   filter2D(input, output2, ddepth, kernel2, anchor, delta, BORDER_DEFAULT );

   output = output1 + output2;

   output.convertTo(output, CV_32F, 1.0 / 255, 0);
   output*=255;

   int64 t1 = cv::getTickCount();
   double secs = (t1-t0)/cv::getTickFrequency();

   cout<< "\nProcessing time on CPU (ms): " << secs*1000 << "\n";   
}













