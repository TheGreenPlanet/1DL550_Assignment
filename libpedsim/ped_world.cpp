#include "ped_world.h"
#include <cassert>

constexpr uint32_t NUMBER_OF_REGIONS = 4;

namespace Ped {
    World::World(const std::vector<Tagent*>& agents) {
        this->regions.push_back(std::make_unique<Region>(0, Region::Bound({0,80}, {0,60})));
        this->regions.push_back(std::make_unique<Region>(1, Region::Bound({80,160}, {0,60})));
        this->regions.push_back(std::make_unique<Region>(2, Region::Bound({0,80}, {60,120})));
        this->regions.push_back(std::make_unique<Region>(3, Region::Bound({80,160}, {60,120})));

        for (const auto& agent : agents) {
            for (const auto& region : this->regions) {
                if (region->contains(agent)) {
                    region->addAgent(agent);
                    break;
                }
            }
        }
    }

    const std::unique_ptr<Region>& World::getRegion(const int regionId) {
        const auto& region = this->regions.at(regionId);
        //assert(region.id == regionId);
        return region;
    }
};