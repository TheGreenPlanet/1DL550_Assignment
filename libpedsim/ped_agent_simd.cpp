#include "ped_agent_simd.h"


namespace Ped {
    TagentSimd::TagentSimd(std::vector<Tagent*> agentsInScenario, const uint32_t instructionsPerTick) {
        // for every 4th agent, initilize a data vector __m128i and push it to the list.
        for (auto i = 0u; i < agentsInScenario.size(); i += instructionsPerTick) {
            __m128i x, y, desiredPositionX, desiredPositionY;
            
            _mm_insert_epi32(x, agentsInScenario[i]->getX(), 0);
            _mm_insert_epi32(x, agentsInScenario[i+1]->getX(), 1);
            _mm_insert_epi32(x, agentsInScenario[i+2]->getX(), 2);
            _mm_insert_epi32(x, agentsInScenario[i+3]->getX(), 3);

            _mm_insert_epi32(y, agentsInScenario[i]->getY(), 0);
            _mm_insert_epi32(y, agentsInScenario[i+1]->getY(), 1);
            _mm_insert_epi32(y, agentsInScenario[i+2]->getY(), 2);
            _mm_insert_epi32(y, agentsInScenario[i+3]->getY(), 3);
            
            _mm_insert_epi32(desiredPositionX, agentsInScenario[i]->getDesiredX(), 0);
            _mm_insert_epi32(desiredPositionX, agentsInScenario[i+1]->getDesiredX(), 1);
            _mm_insert_epi32(desiredPositionX, agentsInScenario[i+2]->getDesiredX(), 2);
            _mm_insert_epi32(desiredPositionX, agentsInScenario[i+3]->getDesiredX(), 3);
            
            _mm_insert_epi32(desiredPositionY, agentsInScenario[i]->getDesiredY(),0);
            _mm_insert_epi32(desiredPositionY, agentsInScenario[i+1]->getDesiredY(), 1);
            _mm_insert_epi32(desiredPositionY, agentsInScenario[i+2]->getDesiredY(), 2);
            _mm_insert_epi32(desiredPositionY, agentsInScenario[i+3]->getDesiredY(), 3);

            this->x.push_back(x);
            this->y.push_back(y);
            this->desiredPositionX.push_back(desiredPositionX);
            this->desiredPositionY.push_back(desiredPositionY);
        }
    }
};