#include "Config.hpp"
#include "VecExtensions.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <zensim/ZpcBuiltin.hpp>
#include <zensim/container/TileVector.hpp>
#include <zensim/container/Vector.hpp>
#include <zensim/cuda/execution/ExecutionPolicy.cuh>
#include <zensim/omp/execution/ExecutionPolicy.hpp>

using vec3 = zs::vec<float, 3>;
using vec4i = zs::vec<int, 4>;
using mat3 = zs::vec<float, 3, 3>;

class Softbody {

    friend class Interface;

    float lambda, mu; //

    std::vector<zs::PropertyTag> vertexTags; //

    int xTag, vTag, aTag; //

    // properties
    zs::TileVector<vec3> d_vertexprops; //
    zs::Vector<vec4i> d_tetras;         //

    zs::Vector<mat3> d_rest;     //
    zs::Vector<float> d_restDet; //
    zs::Vector<mat3> d_restT;    //
    zs::Vector<mat3> d_F;        //

    /*
     * REST
     * det of REST
     * transpose of REST
     * DEFORMED
     * F (deformation grad)
     *
     *
     * */

    size_t numVertex, numTetra; //

    // policies
    zs::OmpExecutionPolicy ompPolicy;   //
    zs::CudaExecutionPolicy cudaPolicy; //

  public:
    Softbody(float _lambda, float _mu, const std::vector<glm::vec3> &_vertices,
             const std::vector<glm::ivec4> &_tetras)
        : lambda(_lambda), mu(_mu) {

        // metadata
        numVertex = _vertices.size();
        numTetra = _tetras.size();
        //        fmt::print("NV = {}, NT = {}", numVertex, numTetra);
        ompPolicy = zs::omp_exec();
        cudaPolicy = zs::cuda_exec();

        // init vertices
        vertexTags = {{"x", 1}, {"v", 1}, {"a", 1}};

        zs::TileVector<vec3> vertexprops{vertexTags, numVertex,
                                         zs::memsrc_e::host};

        xTag = vertexprops.getPropertyOffset("x");
        vTag = vertexprops.getPropertyOffset("v");
        aTag = vertexprops.getPropertyOffset("a");

        // init vertex properties
        ompPolicy(
            zs::range(numVertex),
            [vertexprops = zs::view<zs::execspace_e::openmp>({}, vertexprops),
             &_vertices, xTag = xTag, vTag = vTag,
             aTag = aTag](auto i) mutable {
                vertexprops(xTag, i) =
                    vec3{_vertices[i][0], _vertices[i][1], _vertices[i][2]};

                vertexprops(vTag, i) = vec3::zeros();

                vertexprops(aTag, i) = vec3::zeros();
            });

        d_vertexprops = vertexprops.clone({zs::memsrc_e::device, 0});
        fmt::print("vertices clone\n");

        // init tetra
        // 1. indices
        zs::Vector<vec4i> tetras(numTetra);
        for (int i = 0; i < numTetra; ++i) {

            // 1-inedxed in .node and .ele files
            tetras.setVal(vec4i{_tetras[i][0] - 1, _tetras[i][1] - 1,
                                _tetras[i][2] - 1, _tetras[i][3] - 1},
                          i);
        }

        // fixme: segfault here ? why

        fmt::print("tetras ind init\n");

        d_tetras = tetras.clone({zs::memsrc_e::device, 0});
        fmt::print("tetras clone\n");

        // 2. props
        ComputeAndCloneRestMatrix(vertexprops, tetras);

        d_F = zs::Vector<mat3>(numTetra, zs::memsrc_e::device);
        fmt::print("tetras done");

//        std::exit(42);
    }

    void ComputeAndCloneRestMatrix(zs::TileVector<vec3> &vertexprops,
                                   zs::Vector<vec4i> &tetras) {

        zs::Vector<mat3> rest(numTetra);
        zs::Vector<float> restDet(numTetra);
        zs::Vector<mat3> restT(numTetra);

        ompPolicy(zs::range(numTetra),
                  [vertexprops = zs::view<zs::execspace_e::openmp>(vertexprops),
                   xTag = xTag, rest = zs::view<zs::execspace_e::openmp>(rest),
                   restT = zs::view<zs::execspace_e::openmp>(restT),
                   restDet = zs::view<zs::execspace_e::openmp>(restDet),
                   tetras = zs::view<zs::execspace_e::openmp>(tetras)](
                      auto i) mutable {
                      vec4i elem = tetras(i);
                      // ?
                      vec3 a = vertexprops(xTag, elem[0]);
                      vec3 b = vertexprops(xTag, elem[1]);
                      vec3 c = vertexprops(xTag, elem[2]);
                      vec3 d = vertexprops(xTag, elem[3]);

                      mat3 _restInv = vex::cols(b - a, c - a, d - a);
                      mat3 _rest = zs::inverse(_restInv);
                      rest(i) = _rest;
                      restT(i) = _rest.transpose();
                      restDet(i) = zs::determinant(_rest);
                  });

        d_rest = rest.clone({zs::memsrc_e::device, 0});
        d_restT = restT.clone({zs::memsrc_e::device, 0});
        d_restDet = restDet.clone({zs::memsrc_e::device, 0});
    }

    // --- executed every frame: -------

    void InitFrame() {
        cudaPolicy(
            zs::range(numVertex),
            [d_vertexprops = zs::view<zs::execspace_e::cuda>(d_vertexprops),
             aTag = aTag] __device__(auto i) mutable {
                d_vertexprops(aTag, i) = g;
            });
    }

    void ComputeDeformationGrad() {
        cudaPolicy(
            zs::range(numTetra),
            [vertexprops = zs::view<zs::execspace_e::cuda>(d_vertexprops),
             xTag = xTag, rest = zs::view<zs::execspace_e::cuda>(d_rest),
             F = zs::view<zs::execspace_e::cuda>(d_F),
             tetras = zs::view<zs::execspace_e::cuda>(
                 d_tetras)] __device__(auto i) mutable {
                vec4i elem = tetras(i);
                // ?
                vec3 a = vertexprops(xTag, elem[0]);
                vec3 b = vertexprops(xTag, elem[1]);
                vec3 c = vertexprops(xTag, elem[2]);
                vec3 d = vertexprops(xTag, elem[3]);

                mat3 _deformed = vex::cols(b - a, c - a, d - a);
                mat3 _F = _deformed * rest(i);
                F(i) = _F;
            }

        );
    }

    void ComputeStressForce() {
        cudaPolicy(
            zs::range(numTetra),
            [vertexprops = zs::view<zs::execspace_e::cuda>(d_vertexprops),
             aTag = aTag, rest = zs::view<zs::execspace_e::cuda>(d_rest),
             mu = mu, lambda = lambda, F = zs::view<zs::execspace_e::cuda>(d_F),
             restDet = zs::view<zs::execspace_e::cuda>(d_restDet),
             restT = zs::view<zs::execspace_e::cuda>(d_restT),
             tetras = zs::view<zs::execspace_e::cuda>(
                 d_tetras)] __device__(auto i) mutable {
                mat3 _F = F(i);

                float J = zs::determinant(_F);
                mat3 _F_invT = zs::inverse(_F).transpose();
                mat3 PKStress =
                    mu * (_F - _F_invT) +
                    lambda * static_cast<float>(zs::log(max(J, 0.3))) * _F_invT;

                mat3 forceCols = -1 / (6.f * restDet(i)) * PKStress * restT(i);

                vec3 f1 = zs::col(forceCols, 0);
                vec3 f2 = zs::col(forceCols, 1);
                vec3 f3 = zs::col(forceCols, 2);
                vec3 f0 = -f1 - f2 - f3;

                vec4i elem = tetras(i);
                // atomic add forces(acceleration)
                vex::atomic_add_vec3(&(vertexprops(aTag, elem[0])), f0);
                vex::atomic_add_vec3(&(vertexprops(aTag, elem[1])), f1);
                vex::atomic_add_vec3(&(vertexprops(aTag, elem[2])), f2);
                vex::atomic_add_vec3(&(vertexprops(aTag, elem[3])), f3);
            }

        );
    }

    void ApplyDynamics() {
        cudaPolicy(
            zs::range(numVertex),
            [vertexprops = zs::view<zs::execspace_e::cuda>(d_vertexprops),
             xTag = xTag, vTag = vTag, aTag = aTag] __device__(auto i) mutable {
                vertexprops(vTag, i) += dt * vertexprops(aTag, i);
                vertexprops(vTag, i) *= decay;

                vertexprops(xTag, i) += dt * vertexprops(vTag, i);
            }

        );
    }

    void Collide() {
        cudaPolicy(
            zs::range(numVertex),
            [vertexprops = zs::view<zs::execspace_e::cuda>(d_vertexprops),
             xTag = xTag, vTag = vTag] __device__(auto i) mutable {
                vec3 x = vertexprops(xTag, i);
                if (x[1] < 0) {
                    x[1] = 0;
                    vec3 v = vertexprops(vTag, i);

                    v[1] *= -.9;

                    vertexprops(xTag, i) = x;
                    vertexprops(vTag, i) = v;
                }
            });
    }
};