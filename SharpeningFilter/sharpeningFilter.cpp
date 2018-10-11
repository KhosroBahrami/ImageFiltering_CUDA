//
// C++ implementation of Sharpening Filter
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
extern "C" void sharpeningFilter_CPU (const cv::Mat& input, cv::Mat& output)
{
   Point anchor = Point( -1, -1 );
   double delta = 0;
   int ddepth = -1;
   int kernel_size;

   int64 t0 = cv::getTickCount();

   /// Update kernel size for a normalized box filter
   kernel_size = 3; 

   cv::Mat kernel = (Mat_<double>(kernel_size,kernel_size) << -1, -1, -1, -1, 9, -1, -1, -1, -1);

   // Apply 2D filter to image
   filter2D(input, output, ddepth, kernel, anchor, delta, BORDER_DEFAULT );

   int64 t1 = cv::getTickCount();
   double secs = (t1-t0)/cv::getTickFrequency();

   cout<< "\nProcessing time on CPU (ms): " << secs*1000 << "\n";   
}













