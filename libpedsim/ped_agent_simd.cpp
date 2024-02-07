#include "ped_agent_simd.h"


namespace Ped {
    TagentSimd::TagentSimd(std::vector<Tagent*> agentsInScenario, const uint32_t instructionsPerTick) {
        // for every 4th agent, initilize a data vector __m128i and push it to the list.
        for (auto i = 0u; i < agentsInScenario.size(); i += instructionsPerTick) {
            __m128i x = _mm_setzero_si128();
            __m128i y = _mm_setzero_si128();
            __m128i desiredPositionX = _mm_setzero_si128();
            __m128i desiredPositionY = _mm_setzero_si128();
        
            // Check for out of bounds before accessing agentsInScenario
            if (i < agentsInScenario.size()) {
                x = _mm_insert_epi32(x, agentsInScenario[i]->getX(), 0);
                y = _mm_insert_epi32(y, agentsInScenario[i]->getY(), 0);
                desiredPositionX = _mm_insert_epi32(desiredPositionX, agentsInScenario[i]->getDesiredX(), 0);
                desiredPositionY = _mm_insert_epi32(desiredPositionY, agentsInScenario[i]->getDesiredY(), 0);
            }
            if (i + 1 < agentsInScenario.size()) {
                x = _mm_insert_epi32(x, agentsInScenario[i+1]->getX(), 1);
                y = _mm_insert_epi32(y, agentsInScenario[i+1]->getY(), 1);
                desiredPositionX = _mm_insert_epi32(desiredPositionX, agentsInScenario[i+1]->getDesiredX(), 1);
                desiredPositionY = _mm_insert_epi32(desiredPositionY, agentsInScenario[i+1]->getDesiredY(), 1);
            }
            if (i + 2 < agentsInScenario.size()) {
                x = _mm_insert_epi32(x, agentsInScenario[i+2]->getX(), 2);
                y = _mm_insert_epi32(y, agentsInScenario[i+2]->getY(), 2);
                desiredPositionX = _mm_insert_epi32(desiredPositionX, agentsInScenario[i+2]->getDesiredX(), 2);
                desiredPositionY = _mm_insert_epi32(desiredPositionY, agentsInScenario[i+2]->getDesiredY(), 2);
            }
            if (i + 3 < agentsInScenario.size()) {
                x = _mm_insert_epi32(x, agentsInScenario[i+3]->getX(), 3);
                y = _mm_insert_epi32(y, agentsInScenario[i+3]->getY(), 3);
                desiredPositionX = _mm_insert_epi32(desiredPositionX, agentsInScenario[i+3]->getDesiredX(), 3);
                desiredPositionY = _mm_insert_epi32(desiredPositionY, agentsInScenario[i+3]->getDesiredY(), 3);
            }
        
            this->x.push_back(x);
            this->y.push_back(y);
            this->desiredPositionX.push_back(desiredPositionX);
            this->desiredPositionY.push_back(desiredPositionY);
        }
    }
};