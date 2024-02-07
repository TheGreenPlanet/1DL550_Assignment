//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include <iostream>
#include <stack>
#include <algorithm>
#include <omp.h>
#include <thread>
#include <cmath>
#include <stdlib.h>
#include <cassert>
#include <immintrin.h>
#include <limits>

#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include "cuda_testkernel.h"

namespace Ped {
	constexpr uint32_t SIMD_AGENTS_PER_TICK = 4;

	void Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation)
	{
		// Convenience test: does CUDA work on this machine?
		cuda_test();

		// Set 
		agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());	
		agentsSimd = std::make_unique<Ped::TagentSimd>(agentsInScenario, SIMD_AGENTS_PER_TICK);
		// agentsSimd = new TagentSimd(agentsInScenario);

		// Set up destinations
		destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

		// Sets the chosen implemenation. Standard in the given code is SEQ
		this->implementation = implementation;

		// Set up heatmap (relevant for Assignment 4)
		setupHeatmapSeq();
	}

	void Model::tick()
	{
		if (this->implementation == IMPLEMENTATION::SEQ) {
			for (auto agent : this->getAgents()) {
				agent->computeNextDesiredPosition();
				agent->setX(agent->getDesiredX());
				agent->setY(agent->getDesiredY());
			}
		} else if (this->implementation == IMPLEMENTATION::OMP) {
			//omp_set_num_threads(8);
			#pragma omp parallel for default (none)
			for (auto agent : this->getAgents()) {
				agent->computeNextDesiredPosition();
				agent->setX(agent->getDesiredX());
				agent->setY(agent->getDesiredY());
			}
		} else if (this->implementation == IMPLEMENTATION::PTHREAD) {
			int numThreads = 4;
			std::vector<std::thread> threads(numThreads);
			
			for (int t = 0; t < numThreads; t++) {
				threads[t] = std::thread([t, numThreads](Ped::Model const *model) {
					int start = (model->getAgents().size() * t) / numThreads;
					int end = (model->getAgents().size() * (t + 1)) / numThreads;

					for (int i = start; i < end; i++) {
						Ped::Tagent *agent = model->getAgents()[i];
						agent->computeNextDesiredPosition();
						agent->setX(agent->getDesiredX());
						agent->setY(agent->getDesiredY());
					} }, this);
			}

			for (auto &thread : threads) {
				thread.join();
			}
		} else if (this->implementation == IMPLEMENTATION::VECTOR) {
			auto totalAgents = this->getAgents().size();
			auto totalSize = (totalAgents + SIMD_AGENTS_PER_TICK - 1) / SIMD_AGENTS_PER_TICK; // Calculate needed size, rounded up

			__m256d* nextDestinationsX = (__m256d*)_mm_malloc(totalSize * sizeof(__m256d), 32);
			__m256d* nextDestinationsY = (__m256d*)_mm_malloc(totalSize * sizeof(__m256d), 32);

			for (auto i = 0u; i < totalAgents; i += SIMD_AGENTS_PER_TICK) {
				auto lambda = [&](Twaypoint *waypoint, int agentIdx) -> std::pair<double, double> {
						if (waypoint == NULL) {
								//std::cout << "Agent " << agentIdx << " has no destination" << std::endl;
								constexpr double SOME_BIG_NUMBER_THAT_DOESNT_OVERFLOW = cbrt(std::numeric_limits<double>::max());
								return {SOME_BIG_NUMBER_THAT_DOESNT_OVERFLOW, SOME_BIG_NUMBER_THAT_DOESNT_OVERFLOW};    
						}
						//std::cout << "Agent " << agentIdx << " has destination, x = " << waypoint->getx() << ", y = " << waypoint->gety() << std::endl;
						return {waypoint->getx(), waypoint->gety()};
				};

				double dataX[4] = {0.0, 0.0, 0.0, 0.0};
				double dataY[4] = {0.0, 0.0, 0.0, 0.0};
				
				for (int j = 0; j < SIMD_AGENTS_PER_TICK; ++j) {
						if (i + j >= totalAgents) {
								break;
						}
						auto result = lambda(agents[i+j]->getNextDestination(), i+j);
						dataX[j] = result.first;
						dataY[j] = result.second;
				}

				__m256d fourNextDestinationsX = _mm256_loadu_pd(dataX);
				__m256d fourNextDestinationsY = _mm256_loadu_pd(dataY);
				
				nextDestinationsX[i / SIMD_AGENTS_PER_TICK] = fourNextDestinationsX;
				nextDestinationsY[i / SIMD_AGENTS_PER_TICK] = fourNextDestinationsY;
			}

			assert(agentsSimd->x.size() == agentsSimd->y.size());
			assert(agentsSimd->x.size() == agentsSimd->desiredPositionX.size());
			assert(agentsSimd->x.size() == agentsSimd->desiredPositionY.size());

			const auto simdListSize = agentsSimd->x.size();
			// for the last data vector this might perform some unnecessary calculations
			// but this doesn't affect the actual program since we only extract all valid agents in the loop below this one
			for (auto i = 0u; i < simdListSize; i++) {
				__m128i x = agentsSimd->x.at(i);
				__m128i y = agentsSimd->y.at(i);
				__m256d nextDestinationX = nextDestinationsX[i];
				__m256d nextDestinationY = nextDestinationsY[i];

				// Compute the next desired position
				__m256d diffX = _mm256_sub_pd(nextDestinationX, _mm256_cvtepi32_pd(x));
				__m256d diffY = _mm256_sub_pd(nextDestinationY, _mm256_cvtepi32_pd(y));
				__m256d len = _mm256_sqrt_pd(_mm256_add_pd(_mm256_mul_pd(diffX, diffX), _mm256_mul_pd(diffY, diffY)));

				__m128i desiredPositionXNew = _mm256_cvttpd_epi32(_mm256_add_pd(_mm256_cvtepi32_pd(x), _mm256_div_pd(diffX, len)));
				__m128i desiredPositionYNew = _mm256_cvttpd_epi32(_mm256_add_pd(_mm256_cvtepi32_pd(y), _mm256_div_pd(diffY, len)));

				// std::cout << "Desired position x: " << _mm_extract_epi32(desiredPositionXNew, 0) << std::endl;
				// std::cout << "Desired position y: " << _mm_extract_epi32(desiredPositionYNew, 0) << std::endl;

				// set the desired position
				agentsSimd->desiredPositionX[i] = desiredPositionXNew;
				agentsSimd->desiredPositionY[i] = desiredPositionYNew;

				// set x and y
				agentsSimd->x[i] = desiredPositionXNew;
				agentsSimd->y[i] = desiredPositionYNew;

			}

			// Set the agents' position
			for (auto i = 0u; i < simdListSize; i++) {
				if (i * SIMD_AGENTS_PER_TICK < totalAgents) {
					agents[i * SIMD_AGENTS_PER_TICK]->setX(_mm_extract_epi32(agentsSimd->desiredPositionX[i], 0));
					agents[i * SIMD_AGENTS_PER_TICK]->setY(_mm_extract_epi32(agentsSimd->desiredPositionY[i], 0));
				}
				if ((i * SIMD_AGENTS_PER_TICK + 1) < totalAgents) {
					agents[i * SIMD_AGENTS_PER_TICK + 1]->setX(_mm_extract_epi32(agentsSimd->desiredPositionX[i], 1));
					agents[i * SIMD_AGENTS_PER_TICK + 1]->setY(_mm_extract_epi32(agentsSimd->desiredPositionY[i], 1));
				}
				if ((i * SIMD_AGENTS_PER_TICK + 2) < totalAgents) {
					agents[i * SIMD_AGENTS_PER_TICK + 2]->setX(_mm_extract_epi32(agentsSimd->desiredPositionX[i], 2));
					agents[i * SIMD_AGENTS_PER_TICK + 2]->setY(_mm_extract_epi32(agentsSimd->desiredPositionY[i], 2));
				}
				if ((i * SIMD_AGENTS_PER_TICK + 3) < totalAgents) {
					agents[i * SIMD_AGENTS_PER_TICK + 3]->setX(_mm_extract_epi32(agentsSimd->desiredPositionX[i], 3));
					agents[i * SIMD_AGENTS_PER_TICK + 3]->setY(_mm_extract_epi32(agentsSimd->desiredPositionY[i], 3));
				}
			}
			_mm_free(nextDestinationsX);
			_mm_free(nextDestinationsY);
		}
	}

	////////////
	/// Everything below here relevant for Assignment 3.
	/// Don't use this for Assignment 1!
	///////////////////////////////////////////////

	// Moves the agent to the next desired position. If already taken, it will
	// be moved to a location close to it.
	void Model::move(Ped::Tagent *agent)
	{
		// Search for neighboring agents
		set<const Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

		// Retrieve their positions
		std::vector<std::pair<int, int> > takenPositions;
		for (std::set<const Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
			std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
			takenPositions.push_back(position);
		}

		// Compute the three alternative positions that would bring the agent
		// closer to his desiredPosition, starting with the desiredPosition itself
		std::vector<std::pair<int, int> > prioritizedAlternatives;
		std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
		prioritizedAlternatives.push_back(pDesired);

		int diffX = pDesired.first - agent->getX();
		int diffY = pDesired.second - agent->getY();
		std::pair<int, int> p1, p2;
		if (diffX == 0 || diffY == 0)
		{
			// Agent wants to walk straight to North, South, West or East
			p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
			p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
		}
		else {
			// Agent wants to walk diagonally
			p1 = std::make_pair(pDesired.first, agent->getY());
			p2 = std::make_pair(agent->getX(), pDesired.second);
		}
		prioritizedAlternatives.push_back(p1);
		prioritizedAlternatives.push_back(p2);

		// Find the first empty alternative position
		for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

			// If the current position is not yet taken by any neighbor
			if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

				// Set the agent's position 
				agent->setX((*it).first);
				agent->setY((*it).second);

				break;
			}
		}
	}

	/// Returns the list of neighbors within dist of the point x/y. This
	/// can be the position of an agent, but it is not limited to this.
	/// \date    2012-01-29
	/// \return  The list of neighbors
	/// \param   x the x coordinate
	/// \param   y the y coordinate
	/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
	set<const Tagent*> Model::getNeighbors(int x, int y, int dist) const {

		// create the output list
		// ( It would be better to include only the agents close by, but this programmer is lazy.)	
		return set<const Tagent*>(agents.begin(), agents.end());
	}

	void Model::cleanup() {
		// Nothing to do here right now. 
	}

	Model::~Model()
	{
		std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
		std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
	}
};