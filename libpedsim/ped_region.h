#pragma once
#include <vector>
#include <mutex>
#include <tuple>
#include "ped_agent.h"

namespace Ped {
    class Region
    {
    public:
        const int id;

        struct Bound
        {
            Bound(const std::tuple<int,int>& xSpan, const std::tuple<int,int>& ySpan) : xSpan(xSpan), ySpan(ySpan) {}

            // 160x160
            // x=[0 -> 80) y=[0 -> 80)
            // x=[80 -> 160) y=[0 -> 80)
            // x=[0 -> 80) y=[80 -> 160)
            // x=[80 -> 160) y=[80 -> 160)
            std::tuple<int,int> xSpan; // (0, 80] => 0,1,2...78,79 
            std::tuple<int,int> ySpan; // (0, 80] => 0,1,2...78,79 
        };

        Region(const int id, const Bound& bound) : id(id), bound(bound) {}
        ~Region() = default;

        bool contains(const Tagent* agent) const;
        void lockRegion();
        void unlockRegion();

        void addAgent(Tagent* agent);
        void removeAgent(Tagent* agent);

        std::vector<Tagent*> agents;
    private:        
        std::mutex mutex;
        Bound bound;
    };
};