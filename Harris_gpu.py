#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
Harris conner dection on Nvidia GPU using PyCUDA vs OpenCV python version on CPU.

Ryan Medick
"""

import numpy as np
import cv2
import sys
import time
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit

kernel_code_template = """
__global__ void HarrisCornerDetect(float* In,  float* Sobel_x, float* Sobel_y, float* Out)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    // Sobel X
    if (i < %(MATRIX_SIZE_R)s-1 && j < %(MATRIX_SIZE_C)s-1 && i > 0 && j > 0)
        Sobel_x[i * %(MATRIX_SIZE_C)s + j] = In[(i-1) * %(MATRIX_SIZE_C)s + (j-1)] * -1 + In[(i-1) * %(MATRIX_SIZE_C)s + (j+1)] 
                                           + In[i * %(MATRIX_SIZE_C)s + (j-1)] * -2 + In[i * %(MATRIX_SIZE_C)s + (j+1)]* 2
                                           + In[(i+1) * %(MATRIX_SIZE_C)s + (j-1)] * -1 + In[(i+1) * %(MATRIX_SIZE_C)s + (j+1)];
    else
        Sobel_x[i * %(MATRIX_SIZE_C)s + j] = 0.0; // make sure the board is mark as 0

    __syncthreads();

    // Sobel Y
    if (i < %(MATRIX_SIZE_R)s-1 && j < %(MATRIX_SIZE_C)s-1 && i > 0 && j > 0)
        Sobel_y[i * %(MATRIX_SIZE_C)s + j] = In[(i-1) * %(MATRIX_SIZE_C)s + (j-1)] * -1 + In[(i-1) * %(MATRIX_SIZE_C)s + j] * -2 + In[(i-1) * %(MATRIX_SIZE_C)s + (j+1)] * -1
                                           + In[(i+1) * %(MATRIX_SIZE_C)s + (j-1)]  + In[(i+1) * %(MATRIX_SIZE_C)s + j] * 2 + In[(i+1) * %(MATRIX_SIZE_C)s + (j+1)];
    else
        Sobel_y[i * %(MATRIX_SIZE_C)s + j] = 0.0; // make sure the board is mark as 0

    __syncthreads();

    if (i < %(MATRIX_SIZE_R)s-1 && j < %(MATRIX_SIZE_C)s-1 && i > 0 && j > 0) {
        // Intensity
        float Ixx =  Sobel_x[i * %(MATRIX_SIZE_C)s + j] * Sobel_x[i * %(MATRIX_SIZE_C)s + j];
        float Ixy =  Sobel_x[i * %(MATRIX_SIZE_C)s + j] * Sobel_y[i * %(MATRIX_SIZE_C)s + j]; // Ixy = Iyx
        float Iyy =  Sobel_y[i * %(MATRIX_SIZE_C)s + j] * Sobel_y[i * %(MATRIX_SIZE_C)s + j];

        // Second Moment Matrix, TODO add guassian window function
        float Sxx = Ixx;
        float Sxy = Ixy; // Sxy = Syx
        float Syy = Iyy;

        // Eigen Values
        float EV1 = (Sxx+Syy + sqrt((Sxx+Syy)*(Sxx+Syy) - 4*(Sxx*Syy - Sxy*Sxy))) / 2;
        float EV2 = (Sxx+Syy - sqrt((Sxx+Syy)*(Sxx+Syy) - 4*(Sxx*Syy - Sxy*Sxy))) / 2;

        Out[i * %(MATRIX_SIZE_C)s + j] = EV1 * EV2 - 0.05 * (EV1 + EV2) * (EV1 + EV2);
    }
    else
        Out[i * %(MATRIX_SIZE_C)s + j] = 0.0; // make sure the board is mark as 0;

    __syncthreads();
}
"""


# Load an color image in grayscale
if(len(sys.argv) != 2):
    print("python3 Harris_gpu file")
    exit()

image = cv2.imread(sys.argv[1],0)
if(image is None):
    print("Image not found")
    exit()

gray = image.astype(np.float32)
r,c = gray.shape
print("image is ", r, " by ", c, " pixels")


MATRIX_SIZE_R = r
MATRIX_SIZE_C = c
TILE_SIZE = 2
BLOCK_SIZE = TILE_SIZE

# transfer host (CPU) memory to device (GPU) memory 
in_gpu = gpuarray.to_gpu(gray) 

# create empty gpu array for the results
output_gpu = gpuarray.empty((MATRIX_SIZE_R, MATRIX_SIZE_C), np.float32)
sobel_x_gpu = gpuarray.empty((MATRIX_SIZE_R, MATRIX_SIZE_C), np.float32)
sobel_y_gpu = gpuarray.empty((MATRIX_SIZE_R, MATRIX_SIZE_C), np.float32)

# pass constant data into gpu
kernel_code = kernel_code_template % { 
    'MATRIX_SIZE_R': MATRIX_SIZE_R,
    'MATRIX_SIZE_C': MATRIX_SIZE_C,
    }

# compile the kernel code
mod = compiler.SourceModule(kernel_code)

# get the kernel function from the compiled module
harris = mod.get_function("HarrisCornerDetect")

time1 = time.time()

# call the kernel on the card
harris(
    # inputs
    in_gpu,
    # output
    sobel_x_gpu,
    sobel_y_gpu,
    output_gpu,
    # grid, block info
    grid = (MATRIX_SIZE_R // TILE_SIZE, MATRIX_SIZE_C // TILE_SIZE),
    block = (TILE_SIZE, TILE_SIZE, 1), 
    )

time2 = time.time()

# GPU CUDA
print('GPU took {:.3f} ms'.format((time2-time1)*1000.0))
print("saving Sobel x for image:")
cv2.imwrite('sobel_x.png', sobel_x_gpu.get())
print("saving Sobel y for image:")
cv2.imwrite('sobel_y.png', sobel_y_gpu.get())
print("saving Harris image (GPU) for image:")
cv2.imwrite('gpu_out.png', output_gpu.get())

print("")

# CPU opencv
time1 = time.time()
dst = cv2.cornerHarris(gray,2,3,0.04)
time2 = time.time()
print('CPU took {:.3f} ms'.format((time2-time1)*1000.0))
print("saving Harris image (CPU) for image:")
cv2.imwrite('cpu_out.png', dst)

