//
// Sharpening Filter using CUDA
//
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>

using namespace std;


extern "C" bool sharpeningFilter_GPU_wrapper(const cv::Mat& input, cv::Mat& output);
extern "C" bool sharpeningFilter_CPU(const cv::Mat& input, cv::Mat& output);

// Program main
int main( int argc, char** argv ) {

   // name of image
   string image_name = "sample";

   // input & output file names
   string input_file =  image_name+".jpeg";
   string output_file_cpu = image_name+"_cpu.jpeg";
   string output_file_gpu = image_name+"_gpu.jpeg";

   // Read input image 
   cv::Mat srcImage = cv::imread(input_file ,CV_LOAD_IMAGE_UNCHANGED);
   if(srcImage.empty())
   {
      std::cout<<"Image Not Found: "<< input_file << std::endl;
      return -1;
   }
   cout <<"\ninput image size: "<<srcImage.cols<<" "<<srcImage.rows<<" "<<srcImage.channels()<<"\n";

   // Declare the output image  
   cv::Mat dstImage (srcImage.size(), srcImage.type());

   // run median filter on GPU  
   sharpeningFilter_GPU_wrapper(srcImage, dstImage);
   // Output image
   imwrite(output_file_gpu, dstImage);

   // run median filter on CPU  
   sharpeningFilter_CPU(srcImage, dstImage);
   // Output image
   imwrite(output_file_cpu, dstImage);
      
   return 0;
}





