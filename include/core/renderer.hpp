#pragma once

#include "scene.hpp"
#include "components/mesh.hpp"

namespace DawnViewer {
class Renderer {
 public:
  void init();
  void shutdown();
  void render(Scene& scene);
  void resize(uint32_t width, uint32_t height);

 private:
  wgpu::Instance instance;
  wgpu::Adapter adapter;
  wgpu::Device device;
};
}  // namespace DawnViewer