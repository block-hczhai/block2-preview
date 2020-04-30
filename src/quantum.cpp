
#include <quantum.hpp>

namespace block2 {
    StackAllocator<uint32_t> *ialloc;
    StackAllocator<double> *dalloc;
    DataFrame *frame;
    mt19937 Random::rng;
} // namespace block2