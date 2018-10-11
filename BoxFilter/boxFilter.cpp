//
// C++ implementation of Box Filter
//
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>

#define BLOCK_SIZE      16
#define FILTER_WIDTH    3       
#define FILTER_HEIGHT   3       

using namespace std;
using namespace cv;
struct timeval t1, t2;

// The wrapper is used to call boxFilter 
extern "C" void boxFilter_CPU(const cv::Mat& input, cv::Mat& output)
{
 
   int64 t0 = cv::getTickCount();
   for ( int i = 1; i < 5; i = i + 2 )
   { 
       blur( input, output, Size( i, i ), Point(-1,-1) );
   }

   int64 t1 = cv::getTickCount();
   double secs = (t1-t0)/cv::getTickFrequency();

   cout<< "\nProcessing time for CPU (ms): " << secs*1000 << "\n";   
}













