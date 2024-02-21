#include "ped_region.h"


namespace Ped
{
    bool Region::contains(const Tagent* agent) const
    {
        const auto x = agent->getX();
        const auto y = agent->getY();

        return (x >= std::get<0>(this->bound.xSpan) && x < std::get<1>(this->bound.xSpan) &&
                y >= std::get<0>(this->bound.ySpan) && y < std::get<1>(this->bound.ySpan));
    }

    void Region::lockRegion() {
        this->mutex.lock();
    }

    void Region::unlockRegion() {
        this->mutex.unlock();
    }

    void Region::addAgent(const Tagent *agent) {
        this->lockRegion();
        this->agents.push_back(agent);
        this->unlockRegion();        
    }

    void Region::removeAgent(const Tagent *agent) {
        this->lockRegion();
        // TODO: optimize
        for (auto it = this->agents.begin(); it != this->agents.end(); ++it) {
            if (*it == agent) {
                this->agents.erase(it);
                break;
            }
        }
        this->unlockRegion();
    }
};