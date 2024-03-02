
namespace vex {
using mat3 = zs::vec<float, 3, 3>;

__host__ __device__ mat3 cols(vec3 c1, vec3 c2, vec3 c3) {

    mat3 ret;
    ret[0][0] = c1[0];
    ret[0][1] = c2[0];
    ret[0][2] = c3[0];
    ret[1][0] = c1[1];
    ret[1][1] = c2[1];
    ret[1][2] = c3[1];
    ret[2][0] = c1[2];
    ret[2][1] = c2[2];
    ret[2][2] = c3[2];
    return ret;
}

__device__ void atomic_add_vec3(void *vptr, const vec3 &vec) {

    float *ptr = static_cast<float *>(vptr);

    zs::atomic_add(zs::exec_cuda, ptr, vec[0]);
    zs::atomic_add(zs::exec_cuda, ptr + 1, vec[1]);
    zs::atomic_add(zs::exec_cuda, ptr + 2, vec[2]);
}

} // namespace vex
