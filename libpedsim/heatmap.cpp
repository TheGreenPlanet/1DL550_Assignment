#include "ped_model.h"
#include "chrono"
#include "cuda_runtime.h"
#include "heatmap.cuh"
#include "device_launch_parameters.h"

#include <omp.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <cstdlib>

// Returns the size of the scaled heatmap
int Ped::Model::getHeatmapSize() const {
  // Return the size of the scaled heatmap side length
  return HEATMAP_SCALED_SIDE_LEN;
}

void Ped::Model::processHeatmapUpdates() {
    // Fade heatmap by reducing its intensity
	//auto start = std::chrono::high_resolution_clock::now();
	fadeHeatmap(linear_heatmap);
	//auto end = std::chrono::high_resolution_clock::now();
	//auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	//std::cout << "Heat intensity update time: " << elapsed.count() << "ms" << std::endl;

    // Prepare for heat intensity update based on agent positions
	std::vector<int> x(agents.size()), y(agents.size());
	for (size_t i = 0; i < agents.size(); i++) {
		Ped::Tagent* agent = agents[i];
        // Store x coordinates of agents
		x[i] = agent->getDesiredX();
        // Store y coordinates of agents
		y[i] = agent->getDesiredY();
	}

	//auto start = std::chrono::high_resolution_clock::now();
	// Update heatmap intensity based on agent positions
	updateHeatIntensity(linear_heatmap, &x[0], &y[0], agents.size());
	//auto end = std::chrono::high_resolution_clock::now();
	//auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	//std::cout << "Heat intensity update time: " << elapsed.count() << "ms" << std::endl;

	// Update scaled heatmap and blurred heatmap based on linear heatmap

	//start = std::chrono::high_resolution_clock::now();
	setMaxHeat(linear_heatmap);
	//end = std::chrono::high_resolution_clock::now();
	//elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	//std::cout << "Set max heat time: " << elapsed.count() << "ms" << std::endl;

	//start = std::chrono::high_resolution_clock::now();
	updateScaledHeatmap(linear_heatmap, linear_scaled_heatmap);
	//end = std::chrono::high_resolution_clock::now();
	//elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	//std::cout << "Update scaled heatmap time: " << elapsed.count() << "ms" << std::endl;

	//start = std::chrono::high_resolution_clock::now();
	updateBlurredHeatmap(linear_scaled_heatmap, linear_blurred_heatmnap);
	//end = std::chrono::high_resolution_clock::now();
	//elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	//std::cout << "Update blurred heatmap time: " << elapsed.count() << "ms" << std::endl;

    // Wait for all CUDA operations to complete
	cudaDeviceSynchronize();
}

void Ped::Model::initializeHeatmaps() {
	__shared__ linear_heatmap = (int*)calloc(HEATMAP_SIDE_LEN * HEATMAP_SIDE_LEN, sizeof(int));
	__shared__ linear_scaled_heatmap = (int*)calloc(HEATMAP_SCALED_SIDE_LEN * HEATMAP_SCALED_SIDE_LEN, sizeof(int));
	__shared__ linear_blurred_heatmnap = (int*)calloc(HEATMAP_SCALED_SIDE_LEN * HEATMAP_SCALED_SIDE_LEN, sizeof(int));

	// Allocate pinned host memory for heatmap
	cudaHostAlloc((void**)&linear_heatmap, HEATMAP_SIDE_LEN * HEATMAP_SIDE_LEN * sizeof(int), cudaHostAllocDefault);
    // Allocate pinned host memory for scaled heatmap
	cudaHostAlloc((void**)&linear_scaled_heatmap, HEATMAP_SCALED_SIDE_LEN * HEATMAP_SCALED_SIDE_LEN * sizeof(int), cudaHostAllocDefault);
    // Allocate pinned host memory for blurred heatmap
	cudaHostAlloc((void**)&linear_blurred_heatmnap, HEATMAP_SCALED_SIDE_LEN * HEATMAP_SCALED_SIDE_LEN * sizeof(int), cudaHostAllocDefault);

    // Allocate host memory for pointers to rows of heatmap
	heatmap = (int**)malloc(HEATMAP_SIDE_LEN * sizeof(int*));
    // Allocate host memory for pointers to rows of scaled heatmap
	scaled_heatmap = (int**)malloc(HEATMAP_SCALED_SIDE_LEN * sizeof(int*));
    // Allocate host memory for pointers to rows of blurred heatmap
	blurred_heatmap = (int**)malloc(HEATMAP_SCALED_SIDE_LEN * sizeof(int*));

    // Parallelize row pointer initialization for all heatmaps
	#pragma omp parallel for
	for (int rowIndex = 0; rowIndex < HEATMAP_SCALED_SIDE_LEN; rowIndex++) {
        // Initialize row pointers for heatmap
		if(rowIndex < HEATMAP_SIDE_LEN) {
			heatmap[rowIndex] = linear_heatmap + HEATMAP_SIDE_LEN * rowIndex;
		}
        // Initialize row pointers for scaled heatmap
		scaled_heatmap[rowIndex] = linear_scaled_heatmap + HEATMAP_SCALED_SIDE_LEN * rowIndex;
        // Initialize row pointers for blurred heatmap
		blurred_heatmap[rowIndex] = linear_blurred_heatmnap + HEATMAP_SCALED_SIDE_LEN * rowIndex;
	}
}
