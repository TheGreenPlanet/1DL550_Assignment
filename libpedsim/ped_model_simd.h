#pragma once
#include <nmmintrin.h>

namespace Ped {
    class TagentSimd {
    public:
        TagentSimd(int x, int y, int desiredPositionX, int desiredPositionY);
    private:
        __m128i x;
        __m128i y;
        __m128i desiredPositionX;
        __m128i desiredPositionY;
    };
};