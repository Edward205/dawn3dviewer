#include <dawn/webgpu_cpp.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#define TINYOBJLOADER_IMPLEMENTATION
#include "../tinyobjloader/include/tiny_obj_loader.hpp"

namespace DawnViewer {
inline bool loadMeshFromObj(const std::filesystem::path& path,
                            std::vector<float>& pointData) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string warn;
  std::string err;

  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                              path.string().c_str());

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
  const auto& shape = shapes[0];  // look at the first shape only

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
}  // namespace DawnViewer