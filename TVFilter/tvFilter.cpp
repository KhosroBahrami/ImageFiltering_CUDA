//
// C++ implementation of total variation filter
//
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>

using namespace std;
using namespace cv;
struct timeval t1, t2;

// The wrapper is used to call total variation filter 
extern "C" void tvFilter_CPU(const cv::Mat& input, cv::Mat& output)
{
   Point anchor = Point( -1, -1 );
   double delta = 0;
   int ddepth = -1;
   int kernel_size = 3;

   int64 t0 = cv::getTickCount();

   cv::Mat outputi;
   cv::Mat kernel[8];

   kernel[0] = (Mat_<double>(kernel_size,kernel_size) << -1, 0, 0, 0, 1, 0, 0, 0, 0);
   kernel[1] = (Mat_<double>(kernel_size,kernel_size) << 0, -1, 0, 0, 1, 0, 0, 0, 0);
   kernel[2] = (Mat_<double>(kernel_size,kernel_size) << 0, 0, -1, 0, 1, 0, 0, 0, 0);
   kernel[3] = (Mat_<double>(kernel_size,kernel_size) << 0, 0, 0, -1, 1, 0, 0, 0, 0);
   kernel[4] = (Mat_<double>(kernel_size,kernel_size) << 0, 0, 0, 0, 1, -1, 0, 0, 0);
   kernel[5] = (Mat_<double>(kernel_size,kernel_size) << 0, 0, 0, 0, 1, 0, -1, 0, 0);
   kernel[6] = (Mat_<double>(kernel_size,kernel_size) << 0, 0, 0, 0, 1, 0, 0, -1, 0);
   kernel[7] = (Mat_<double>(kernel_size,kernel_size) << 0, 0, 0, 0, 1, 0, 0, 0, -1);
   
   for(int i=0 ; i<8 ; i++){
      // Apply 2D filter
      filter2D(input, outputi, ddepth, kernel[i], anchor, delta, BORDER_DEFAULT );
      output += outputi;
   }
   
   int64 t1 = cv::getTickCount();
   double secs = (t1-t0)/cv::getTickFrequency();

   cout<< "\nProcessing time on CPU (ms): " << secs*1000 << "\n";   
}













