#include <fmt/format.h>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <string>
#include <vector>

class ReadMesh {

    std::string nodePath, elePath;

    void ReadNode(std::ifstream &file, std::vector<glm::vec3> &vertices) {

        //        int numVertex, dim;
        //        int dummy;
        //        file >> numVertex >> dim >> dummy >> dummy;
        //
        //        if (dim != 3) {
        //            fmt::print("dimension must be 3, found {}", dim);
        //            std::exit(1);
        //        }
        //
        //        vertices.resize(0);
        //        vertices.reserve(numVertex);
        //
        //        fmt::print(" >>> Reading Nodes, numVertex = {}", numVertex);
        //
        //
        //        float x, y, z;
        //
        //        for (int i = 0; i < numVertex; ++i) {
        //
        //            file >> dummy >> x >> y >> z >> dummy;
        //
        //            vertices.push_back({x, y, z});
        //        }
    }

    void ReadEle(std::ifstream &file, std::vector<glm::ivec4> &tetras) {

        //        int numTetra, dim;
        //
        //        int dummy;
        //        file >> numTetra >> dim >> dummy;
        //        int q, w, e, r;
        //
        //        if (dim != 4) {
        //            fmt::print("dimension must be 3, found {}", dim);
        //            std::exit(1);
        //        }
        //
        //        tetras.resize(0);
        //        tetras.reserve(numTetra);
        //
        //        fmt::print(" >>> Reading Elements, numTetra = {}", numTetra);
        //
        //        for (int i = 0; i < numTetra; ++i) {
        //            file >> dummy >> q >> w >> e >> r;
        //            tetras.push_back({q, w, e, r});
        //        }
    }

  public:
    ReadMesh Fetch(std::string _nodePath, std::string _elePath) {
        nodePath = _nodePath;
        elePath = _elePath;
        return *this;
    }

    void Read(std::vector<glm::vec3> &vertices,
              std::vector<glm::ivec4> &tetras) {

        //        std::ifstream nodeFile(nodePath);
        //        if (!nodeFile.is_open()) {
        //            throw std::runtime_error("Failed to open .node file: " +
        //            nodePath);
        //        }
        //
        //        std::ifstream eleFile(elePath);
        //        if (!eleFile.is_open()) {
        //            throw std::runtime_error("Failed to open .ele file: " +
        //            elePath);
        //        }
        //
        //        ReadNode(nodeFile, vertices);
        //        ReadEle(eleFile, tetras);

        vertices = {{1, 6, 1}, {6, 6, 1}, {1, 12, 1}, {1, 6, 6}};
        tetras = {{1, 2, 3, 4}};
    }
};