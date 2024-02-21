#pragma once
#include <vector>
#include <memory>

#include "ped_agent.h"
#include "ped_region.h"

namespace Ped {
    class World {
    public:
        World(const std::vector<Tagent*>& agents);

        const std::unique_ptr<Region>& getRegion(const int regionId);
    private:
        std::vector<std::unique_ptr<Region>> regions;
    };
};
