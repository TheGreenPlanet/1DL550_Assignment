#pragma once
#include <immintrin.h>
#include <vector>
#include "ped_agent.h"
#include <stdint.h>

namespace Ped {
    class TagentSimd {
    public:
        TagentSimd(std::vector<Ped::Tagent*> agentsInScenario, const uint32_t instrPerTick);
        ~TagentSimd();
        std::vector<__m128i> x;
        std::vector<__m128i> y;
        std::vector<__m128i> desiredPositionX;
        std::vector<__m128i> desiredPositionY;


		__m256d* nextDestinationsX;
		__m256d* nextDestinationsY;
    };
};