#pragma once
#include <nmmintrin.h>
#include <vector>
#include "ped_agent.h"
#include <emmintrin.h>
#include <stdint.h>

namespace Ped {
    struct TagentSimd {
        TagentSimd(std::vector<Ped::Tagent*> agentsInScenario, const uint32_t instrPerTick);

        std::vector<__m128i> x;
        std::vector<__m128i> y;
        std::vector<__m128i> desiredPositionX;
        std::vector<__m128i> desiredPositionY;
    };
};