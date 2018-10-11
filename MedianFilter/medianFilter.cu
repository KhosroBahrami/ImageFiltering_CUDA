//
// CUDA implementation of Median Filter
//
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"

#define BLOCK_SIZE      16
#define FILTER_WIDTH    3       
#define FILTER_HEIGHT   3       

using namespace std;

// Sort function on device
__device__ void sort(unsigned char* filterVector)
{
	 for (int i = 0; i < FILTER_WIDTH*FILTER_HEIGHT; i++) {
	    for (int j = i + 1; j < FILTER_WIDTH*FILTER_HEIGHT; j++) {
		if (filterVector[i] > filterVector[j]) { 
	              //Swap the variables
		      unsigned char tmp = filterVector[i];
		      filterVector[i] = filterVector[j];
		      filterVector[j] = tmp;
		}
             }
         }
}

// Run Median Filter on GPU
__global__ void medianFilter(unsigned char *srcImage, unsigned char *dstImage, unsigned int width, unsigned int height, int channel)
{
   int x = blockIdx.x*blockDim.x + threadIdx.x;
   int y = blockIdx.y*blockDim.y + threadIdx.y;

   // only threads inside image will write results
   if((x>=FILTER_WIDTH/2) && (x<(width-FILTER_WIDTH/2)) && (y>=FILTER_HEIGHT/2) && (y<(height-FILTER_HEIGHT/2)))
   {
      for(int c=0 ; c<channel ; c++)   
      {
         unsigned char filterVector[FILTER_WIDTH*FILTER_HEIGHT];     
         // Loop inside the filter to average pixel values
         for(int ky=-FILTER_HEIGHT/2; ky<=FILTER_HEIGHT/2; ky++) {
            for(int kx=-FILTER_WIDTH/2; kx<=FILTER_WIDTH/2; kx++) {
               filterVector[ky*FILTER_WIDTH+kx] = srcImage[((y+ky)*width + (x+kx))*channel+c];
            }
         }
         // Sorting values of filter   
         sort(filterVector);
         dstImage[(y*width+x)*channel+c] =  filterVector[(FILTER_WIDTH*FILTER_HEIGHT)/2];
      }
   }
}


// The wrapper to call median filter 
extern "C" void medianFilter_GPU_wrapper(const cv::Mat& input, cv::Mat& output)
{
        // Use cuda event to catch time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Calculate number of image channels
        int channel = input.step/input.cols; 

        // Calculate number of input & output bytes in each block
        const int inputSize = input.cols * input.rows * channel;
        const int outputSize = output.cols * output.rows * channel;
        unsigned char *d_input, *d_output;
        
        // Allocate device memory
        cudaMalloc<unsigned char>(&d_input,inputSize);
        cudaMalloc<unsigned char>(&d_output,outputSize);

        // Copy data from OpenCV input image to device memory
        cudaMemcpy(d_input,input.ptr(),inputSize,cudaMemcpyHostToDevice);

        // Specify block size
        const dim3 block(BLOCK_SIZE,BLOCK_SIZE);

        // Calculate grid size to cover the whole image
        const dim3 grid((output.cols + block.x - 1)/block.x, (output.rows + block.y - 1)/block.y);

        // Start time
        cudaEventRecord(start);

        // Run BoxFilter kernel on CUDA 
        medianFilter<<<grid,block>>>(d_input, d_output, output.cols, output.rows, channel);

        // Stop time
        cudaEventRecord(stop);

        //Copy data from device memory to output image
        cudaMemcpy(output.ptr(),d_output,outputSize,cudaMemcpyDeviceToHost);

        //Free the device memory
        cudaFree(d_input);
        cudaFree(d_output);

        cudaEventSynchronize(stop);
        float milliseconds = 0;
        
        // Calculate elapsed time in milisecond  
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout<< "\nProcessing time on GPU (ms): " << milliseconds << "\n";
}













