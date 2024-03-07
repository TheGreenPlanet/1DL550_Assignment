#include "device_launch_parameters.h"
#include "heatmap.cuh"
#include "ped_model.h"
#include "cuda_runtime.h" 

#include <cstdlib> 
#include <cuda.h>
#include <cmath> 
#include <cuda_runtime_api.h>
#include <omp.h> 

// Kernel to cap the heatmap values at a maximum of 255
__global__ void setMaxHeatKernel(int *d_heatmap) {
    // blockIdx.x is the block index within the grid
    // blockDim.x is the number of threads per block
    // threadIdx.x is the thread index within the block
    // The following calculation is used to determine the global index of the thread, meaning the index in the array
    int index = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global index in the array
    // Check if the index is within the bounds of the heatmap
    if (index < HEATMAP_SIDE_LEN*HEATMAP_SIDE_LEN) {
        // Cap the heatmap value at the index to a maximum of 255
        if (d_heatmap[index] >= 255) {
            d_heatmap[index] = 255;
        }
    }
}

// Kernel to fade the heatmap by reducing each cell's value by 20%
__global__ void fadeHeatmapKernel(int *d_heatmap) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global index in the array
    // Check if the index is within the bounds of the heatmap
    if (index < HEATMAP_SIDE_LEN*HEATMAP_SIDE_LEN) {
        // Reduce the heatmap value at the index by 20%, rounding to nearest integer
        d_heatmap[index] = (int)round(d_heatmap[index] * 0.80);
    }
}

// Kernel to scale up the heatmap for visualization
__global__ void scaleHeatmapKernel(int *d_heatmap, int *d_scaledHeatmap) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global index in the scaled array
    // Check if the index is within the bounds of the scaled heatmap
    if (index < HEATMAP_SCALED_SIDE_LEN*HEATMAP_SCALED_SIDE_LEN) {
        // Loop through cells in the scaled heatmap
        for (int cellY = 0; cellY < HEATMAP_CELLSIZE; cellY++) {
            int scaledBlock = (blockIdx.x * HEATMAP_CELLSIZE + cellY) * blockDim.x * HEATMAP_CELLSIZE;
            for (int cellX = 0; cellX < HEATMAP_CELLSIZE; cellX++) {
                int scaledThread = threadIdx.x * HEATMAP_CELLSIZE + cellX;
                // Copy the heatmap value to the scaled heatmap at the new location
                d_scaledHeatmap[scaledBlock + scaledThread] = d_heatmap[index];
            }
        }
    }
}

// Kernel to blur the scaled heatmap for smoother visualization
__global__ void blurHeatmapKernel(int *d_scaledHeatmap, int *d_blurredHeatmap) {
    int WEIGHTSUM = 273; // Define the sum of weights for the blur kernel
    const int w[5][5] = { // Define the blur kernel weights
        { 1, 4, 7, 4, 1 },
        { 4, 16, 26, 16, 4 },
        { 7, 26, 41, 26, 7 },
        { 4, 16, 26, 16, 4 },
        { 1, 4, 7, 4, 1 }
    };

    // i = The global start index of the block
    int i = blockIdx.x * blockDim.x;
    // modI = The modified index for boundary checks, used to avoid out-of-bounds access
    int modI = (blockIdx.x * blockDim.x + threadIdx.x) % HEATMAP_SCALED_SIDE_LEN;
    // j = The thread index within the block
    int j = threadIdx.x;
    // sum = The sum of the weighted values of the neighbors
    int sum = 0;
    // Perform boundary checks and calculate weighted average for blurring
    // 2 = The distance in cells from the edge of the heatmap where the blur kernel is applied.
    if (modI >= 2 && modI < HEATMAP_SCALED_SIDE_LEN-2 && i+j >= 2*HEATMAP_SCALED_SIDE_LEN && blockIdx.x < 5*HEATMAP_SCALED_SIDE_LEN-2) {
        for (int k = -2; k < 3; k++) { // k moves vertically through the neighbors
            for (int l = -2; l < 3; l++) { // j moves horizontally through the neighbors
                if(i+HEATMAP_SCALED_SIDE_LEN*k >= 2*HEATMAP_SCALED_SIDE_LEN && i+HEATMAP_SCALED_SIDE_LEN*k < HEATMAP_SCALED_SIDE_LEN*HEATMAP_SCALED_SIDE_LEN-2*HEATMAP_SCALED_SIDE_LEN) {
                    // Add the weighted value of the neighbor to the sum
                    sum += w[2 + k][2 + l] * d_scaledHeatmap[(i + k) + (j + l)]; 
                }
            }
        }
    }
    int value = sum / WEIGHTSUM; // Calculate the blurred value by dividing the sum by the weight sum
    d_blurredHeatmap[i + j] = 0x00FF0000 | value << 24; // Set the blurred value with alpha channel for visualization
}


// Kernel to increase heat intensity at specific agent locations
__global__ void heatIntensity(int *d_heatmap, int *x, int *y, int agent_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global index in the array
    // Check if index is within bounds of the agents array
    if (index >= 0 && index < agent_size) {
        // Increase the heatmap value at the agent's position by 40
        atomicAdd(&d_heatmap[y[index]*HEATMAP_SIDE_LEN + x[index]], 40);
    }
}

// Define the number of threads per block for CUDA kernel launches
constexpr std::size_t THREADS_PER_BLOCK = 1024;

// Function to scale up the heatmap for visualization
void updateScaledHeatmap(int *linear_heatmap, int *linear_scaled_heatmap) {
    int scaledSizeSquared = HEATMAP_SIDE_LEN*HEATMAP_SIDE_LEN*HEATMAP_CELLSIZE*HEATMAP_CELLSIZE; // Calculate scaled heatmap size
    int *d_heatmap, *d_scaledHeatmap; // Pointers for device memory (memory on the GPU)

    cudaMalloc((void **)&d_heatmap, HEATMAP_SIDE_LEN*HEATMAP_SIDE_LEN*sizeof(int)); // Allocate device memory for heatmap
    cudaMalloc((void **)&d_scaledHeatmap, scaledSizeSquared*sizeof(int)); // Allocate device memory for scaled heatmap

    // Copy data from host to device
    cudaMemcpyAsync(d_heatmap, linear_heatmap, HEATMAP_SIDE_LEN*HEATMAP_SIDE_LEN*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_scaledHeatmap, linear_scaled_heatmap, scaledSizeSquared*sizeof(int), cudaMemcpyHostToDevice);

    scaleHeatmapKernel<<<(HEATMAP_SIDE_LEN*HEATMAP_SIDE_LEN)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_heatmap, d_scaledHeatmap); // Launch the scaleHeatmap kernel

    // Copy data back to host
    cudaMemcpyAsync(linear_heatmap, d_heatmap, HEATMAP_SIDE_LEN*HEATMAP_SIDE_LEN*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(linear_scaled_heatmap, d_scaledHeatmap, scaledSizeSquared*sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_heatmap);  
    cudaFree(d_scaledHeatmap);  
}

// Function to increase the heat intensity at agent locations
void updateHeatIntensity(int *linear_heatmap, int *x, int *y, int agent_size) {
    int *d_heatmap, *d_x, *d_y; // Pointers for device memory (memory on the GPU)

    cudaMalloc((void **)&d_heatmap, HEATMAP_SIDE_LEN*HEATMAP_SIDE_LEN*sizeof(int)); // Allocate device memory for heatmap
    cudaMalloc((void **)&d_x, agent_size*sizeof(int)); // Allocate device memory for x coordinates
    cudaMalloc((void **)&d_y, agent_size*sizeof(int)); // Allocate device memory for y coordinates

    // Copy data from host to device
    cudaMemcpyAsync(d_heatmap, linear_heatmap, HEATMAP_SIDE_LEN*HEATMAP_SIDE_LEN*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_x, x, agent_size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_y, y, agent_size*sizeof(int), cudaMemcpyHostToDevice);

    heatIntensity<<<((agent_size)/THREADS_PER_BLOCK)+1,THREADS_PER_BLOCK>>>(d_heatmap, d_x, d_y, agent_size); // Launch the heatIntensify kernel

    // Copy data back to host
    cudaMemcpyAsync(linear_heatmap, d_heatmap, HEATMAP_SIDE_LEN*HEATMAP_SIDE_LEN*sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_heatmap);
    cudaFree(d_x);
    cudaFree(d_y);
}

// Function to blur the scaled heatmap for smoother visualization
void updateBlurredHeatmap(int *linear_scaled_heatmap, int *linear_blurred_heatmap) {
    int *d_scaledHeatmap, *d_blurredHeatmap; // Pointers for device memory (memory on the GPU)
    cudaMalloc((void **)&d_scaledHeatmap, HEATMAP_SCALED_SIDE_LEN * HEATMAP_SCALED_SIDE_LEN * sizeof(int)); // Allocate device memory for scaled heatmap
    cudaMalloc((void **)&d_blurredHeatmap, HEATMAP_SCALED_SIDE_LEN * HEATMAP_SCALED_SIDE_LEN * sizeof(int)); // Allocate device memory for blurred heatmap

    // Copy data from host to device
    cudaMemcpyAsync(d_blurredHeatmap, linear_blurred_heatmap, HEATMAP_SCALED_SIDE_LEN * HEATMAP_SCALED_SIDE_LEN * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_scaledHeatmap, linear_scaled_heatmap, HEATMAP_SCALED_SIDE_LEN * HEATMAP_SCALED_SIDE_LEN * sizeof(int), cudaMemcpyHostToDevice);

    blurHeatmapKernel<<<(HEATMAP_SCALED_SIDE_LEN * HEATMAP_SCALED_SIDE_LEN)/ THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_scaledHeatmap, d_blurredHeatmap); // Launch the blurHeatmap kernel

    // Copy data back to host
    cudaMemcpyAsync(linear_blurred_heatmap, d_blurredHeatmap, HEATMAP_SCALED_SIDE_LEN * HEATMAP_SCALED_SIDE_LEN * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(linear_scaled_heatmap, d_scaledHeatmap, HEATMAP_SCALED_SIDE_LEN * HEATMAP_SCALED_SIDE_LEN * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_blurredHeatmap);  
    cudaFree(d_scaledHeatmap);  
}

// Function to update the heatmap by fading the heat values
void fadeHeatmap(int *linear_heatmap) {
    int *d_heatmap; // Pointer for device memory
    cudaMalloc((void **)&d_heatmap, HEATMAP_SIDE_LEN*HEATMAP_SIDE_LEN*sizeof(int)); // Allocate device memory

    cudaMemcpyAsync(d_heatmap, linear_heatmap, HEATMAP_SIDE_LEN*HEATMAP_SIDE_LEN*sizeof(int), cudaMemcpyHostToDevice); // Copy data from host to device

    fadeHeatmapKernel<<<((HEATMAP_SIDE_LEN*HEATMAP_SIDE_LEN)/THREADS_PER_BLOCK)+1,THREADS_PER_BLOCK>>>(d_heatmap); // Launch the fadeHeat kernel

    cudaMemcpyAsync(linear_heatmap, d_heatmap, HEATMAP_SIDE_LEN*HEATMAP_SIDE_LEN*sizeof(int), cudaMemcpyDeviceToHost); // Copy data back to host
    cudaFree(d_heatmap); // Free device memory
}


// Function to cap the heat values at a maximum of 255
void setMaxHeat(int *linear_heatmap) {
    int *d_heatmap; // Pointer for device memory
    cudaMalloc((void **)&d_heatmap, HEATMAP_SIDE_LEN * HEATMAP_SIDE_LEN * sizeof(int)); // Allocate device memory

    cudaMemcpyAsync(d_heatmap, linear_heatmap, HEATMAP_SIDE_LEN * HEATMAP_SIDE_LEN * sizeof(int), cudaMemcpyHostToDevice); // Copy data from host to device

    setMaxHeatKernel<<<((HEATMAP_SIDE_LEN * HEATMAP_SIDE_LEN)/THREADS_PER_BLOCK)+1,THREADS_PER_BLOCK>>>(d_heatmap); // Launch the setMaxHeat kernel

    cudaMemcpyAsync(linear_heatmap, d_heatmap, HEATMAP_SIDE_LEN * HEATMAP_SIDE_LEN * sizeof(int), cudaMemcpyDeviceToHost); // Copy data back to host
    cudaFree(d_heatmap); // Free device memory
}

