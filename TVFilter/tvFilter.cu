//
// CUDA implementation of Total Variation Filter
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

// Run Total Variation Filter on GPU
__global__ void tvFilter(unsigned char *srcImage, unsigned char *dstImage, unsigned int width, unsigned int height)
{
   int x = blockIdx.x*blockDim.x + threadIdx.x;
   int y = blockIdx.y*blockDim.y + threadIdx.y;

   // only threads inside image will write results
   if((x>=FILTER_WIDTH/2) && (x<(width-FILTER_WIDTH/2)) && (y>=FILTER_HEIGHT/2) && (y<(height-FILTER_HEIGHT/2)))
   {
         float sod = 0;
         // Loop inside the filter to average pixel values
         for(int ky=-FILTER_HEIGHT/2; ky<=FILTER_HEIGHT/2; ky++) {
            for(int kx=-FILTER_WIDTH/2; kx<=FILTER_WIDTH/2; kx++) {
               float fl = srcImage[((y+ky)*width + (x+kx))];
               float center = srcImage[((y)*width + (x))];
               sod += fl-center;
            }
         }
         dstImage[(y*width+x)] = sod;
   }
}


// The wrapper is used to call total variation filter 
extern "C" void tvFilter_GPU_wrapper(const cv::Mat& input, cv::Mat& output)
{
        // Use cuda event to catch time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Calculate number of input & output bytes in each block
        const int inputSize = input.cols * input.rows;
        const int outputSize = output.cols * output.rows;
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
        tvFilter<<<grid,block>>>(d_input, d_output, output.cols, output.rows);

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
        cout<< "\nTotal processing time on GPU (ms): " << milliseconds << "\n";
}













