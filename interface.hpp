#include "ReadMesh.hpp"
#include "SoftBody.hpp"
#include "memory"
#include <vector>

class Interface {

    std::vector<glm::ivec4> tetras;
    std::unique_ptr<Softbody> sb;

  public:
    Interface() {}

    void init(std::vector<glm::vec3> &vertices,
              std::vector<glm::ivec3> &indices) {

        // parameters
        float lambda, mu;
        fmt::print("lambda, mu\n");
        std::cin >> lambda >> mu;

        // READ MESH, init vertices and tetra indices
        // ----------------------------
        ReadMesh readMesh;
        readMesh.Fetch("../model/house2.node", "../model/house2.ele")
            .Read(vertices, tetras);
        fmt::print("read done\n");

        // -----------------------------------------------------------------------

        // init triangle indices
        indices.resize(0);
        for (auto &&tetra : tetras) {

            indices.push_back({tetra[0], tetra[1], tetra[2]});
            indices.push_back({tetra[0], tetra[1], tetra[3]});
            indices.push_back({tetra[0], tetra[2], tetra[3]});
            indices.push_back({tetra[1], tetra[2], tetra[3]});
        }

        fmt::print("tetra configured");

        // init softbody
        sb = std::make_unique<Softbody>(lambda, mu, vertices, tetras);
    }

    void update(std::vector<glm::vec3> &vertices) {

        sb->InitFrame();
        //        fmt::print("init frame\n");
        sb->ComputeDeformationGrad();
        //        //        fmt::print("F\n");
        sb->ComputeStressForce();
        //        //        fmt::print("stress&force\n");
        sb->ApplyDynamics();
        //        //        fmt::print("apply\n");
        sb->Collide();
        //        fmt::print("collide\n");

        // copy
        auto h_vertexprops = sb->d_vertexprops.clone({zs::memsrc_e::host, 0});
        //        fmt::print("clone back to host");

        sb->ompPolicy(

            zs::range(sb->numVertex),
            [vertexprops = zs::view<zs::execspace_e::openmp>(h_vertexprops),
             xTag = sb->xTag, &vertices](auto i) mutable {
                //                std::cout << i << std::endl;

                vec3 xi = vertexprops(xTag, i);
                vertices[i] = {xi[0], xi[1], xi[2]};

//                fmt::print("x[{}] = ({}, {}, {})\n\n", i, xi[0], xi[1], xi[2]);
            });
    }
};
