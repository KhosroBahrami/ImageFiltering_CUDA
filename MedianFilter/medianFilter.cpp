//
// C++ implementation of Median Filter
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

// The wrapper is use to call median filter 
extern "C" void medianFilter_CPU (const cv::Mat& input, cv::Mat& output)
{

   cv::Mat input_gray;
   int kernel_size = 3;
   int scale = 1;
   int delta = 0;
   int ddepth = CV_16S;
 
   int64 t0 = cv::getTickCount();

   for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
        medianBlur (input, output, i );


   int64 t1 = cv::getTickCount();
   double secs = (t1-t0)/cv::getTickFrequency();

   cout<< "\nProcessing time on CPU (ms): " << secs*1000 << "\n";   
}













