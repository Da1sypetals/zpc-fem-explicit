
#include <zensim/ZpcBuiltin.hpp>
#include <zensim/container/TileVector.hpp>
#include <zensim/container/Vector.hpp>
using vec3 = zs::vec<float, 3>;


constexpr float dt = 0.01;
constexpr int dim = 3;
constexpr __host__ __device__ vec3 g = {0, -9.8, 0};
constexpr float decay = 0.995;
