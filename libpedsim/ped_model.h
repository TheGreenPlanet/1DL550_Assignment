//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <vector>
#include <map>
#include <set>
#include <memory>

#include "ped_agent.h"
#include "ped_agent_simd.h"
#include "ped_world.h"

constexpr std::size_t HEATMAP_SIDE_LEN = 1024;
constexpr std::size_t HEATMAP_CELLSIZE = 5;
constexpr std::size_t HEATMAP_SCALED_SIDE_LEN = HEATMAP_SIDE_LEN * HEATMAP_CELLSIZE;

constexpr std::size_t SIZE = 1024;
constexpr std::size_t CELLSIZE = 5;
constexpr std::size_t SCALED_SIZE = SIZE * CELLSIZE;

namespace Ped{
	class Tagent;

	// The implementation modes for Assignment 1 + 2:
	// chooses which implementation to use for tick()
	enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ, SEQMOVE, MOVE };
	enum HEATMAP_IMPLEMENTATION { H_SEQ, H_CUDA };

	class Model
	{
	public:

		// Sets everything up
		void setup(std::vector<Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario,IMPLEMENTATION implementation, HEATMAP_IMPLEMENTATION heatmap_implementation);
		// Coordinates a time step in the scenario: move all agents by one step (if applicable).
		void tick();

		// Returns the agents of this scenario
		const std::vector<Tagent*> getAgents() const { return agents; };

		// Adds an agent to the tree structure
		void placeAgent(const Ped::Tagent *a);

		// Cleans up the tree and restructures it. Worth calling every now and then.
		void cleanup();
		~Model();

		// Returns the heatmap visualizing the density of agents
		int const * const * getHeatmap() const { return blurred_heatmap; };
		int getHeatmapSize() const;

		void setupHeatmapSeq();
		void updateHeatmapSeq();

	private:


		// Denotes which implementation (sequential, parallel implementations..)
		// should be used for calculating the desired positions of
		// agents (Assignment 1)
		IMPLEMENTATION implementation;
		
		HEATMAP_IMPLEMENTATION heatmap_implementation;

		// The agents in this scenario
		std::vector<Tagent*> agents;

		// TagentSimd *agentsSimd;
		std::unique_ptr<Ped::TagentSimd> agentsSimd;

		// The waypoints in this scenario
		std::vector<Twaypoint*> destinations;

		// The world
		std::unique_ptr<Ped::World> world;

		// Moves an agent towards its next position
		void move(Ped::Tagent *agent);

		////////////
		/// Everything below here won't be relevant until Assignment 3
		///////////////////////////////////////////////

		// Returns the set of neighboring agents for the specified position
		set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;

		////////////
		/// Everything below here won't be relevant until Assignment 4
		///////////////////////////////////////////////



		// The heatmap representing the density of agents
		int ** heatmap;

		// The scaled heatmap that fits to the view
		int ** scaled_heatmap;

		// The final heatmap: blurred and scaled to fit the view
		int ** blurred_heatmap;


		int *linear_heatmap;
		int *linear_scaled_heatmap;
		int *linear_blurred_heatmnap;

		void initializeHeatmaps();
		void processHeatmapUpdates();
		void updateHeatmapCUDA();
	};
}
#endif
