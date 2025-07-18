#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <dawn/webgpu_cpp.h>
#include <iostream>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.hpp"

inline bool loadGeometry(
    const std::filesystem::path& path,
    std::vector<float>& pointData,
    std::vector<uint16_t>& indexData,
    int dimensions
) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    pointData.clear();
    indexData.clear();

    enum class Section {
        None,
        Points,
        Indices,
    };
    Section currentSection = Section::None;

    float value;
    uint16_t index;
    std::string line;
    while (!file.eof()) {
        getline(file, line);
        
        // overcome the `CRLF` problem
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        if (line == "[points]") {
            currentSection = Section::Points;
        }
        else if (line == "[indices]") {
            currentSection = Section::Indices;
        }
        else if (line[0] == '#' || line.empty()) {
            // Do nothing, this is a comment
        }
        else if (currentSection == Section::Points) {
            std::istringstream iss(line);
            // Get x, y, r, g, b
            for (int i = 0; i < dimensions + 3; ++i) {
                iss >> value;
                pointData.push_back(value);
            }
        }
        else if (currentSection == Section::Indices) {
            std::istringstream iss(line);
            // Get corners #0 #1 and #2
            for (int i = 0; i < 3; ++i) {
                iss >> index;
                indexData.push_back(index);
            }
        }
    }
    return true;
}

bool loadGeometryFromObj(const std::filesystem::path& path, std::vector<float>& pointData) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path.string().c_str());

    if (!warn.empty()) {
        std::cout << warn << std::endl;
    }

    if (!err.empty()) {
        std::cerr << err << std::endl;
    }

    if (!ret) {
        return false;
    }

    // Filling in vertexData:
    const auto& shape = shapes[0]; // look at the first shape only
    
    pointData.clear();
    for (size_t i = 0; i < shape.mesh.indices.size(); ++i) {
        const tinyobj::index_t& idx = shape.mesh.indices[i];

        pointData.push_back(attrib.vertices[3 * idx.vertex_index + 0]);
        pointData.push_back(-attrib.vertices[3 * idx.vertex_index + 2]);
        pointData.push_back(attrib.vertices[3 * idx.vertex_index + 1]);
        pointData.push_back(attrib.normals[3 * idx.normal_index + 0]);
        pointData.push_back(-attrib.normals[3 * idx.normal_index + 2]);
        pointData.push_back(attrib.normals[3 * idx.normal_index + 1]);
        pointData.push_back(attrib.colors[3 * idx.vertex_index + 0]);
        pointData.push_back(attrib.colors[3 * idx.vertex_index + 1]);
        pointData.push_back(attrib.colors[3 * idx.vertex_index + 2]);
        
    }
    return true;
}