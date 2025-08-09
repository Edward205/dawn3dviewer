#pragma once

#include "GLFW/glfw3.h"
#include "webgpu/webgpu_cpp.h"

namespace DawnViewer {
class Window {
 public:
  void create(const char* title, int width, int height);
  wgpu::Surface createSurface(wgpu::Instance instance);
  void shutdown();
  ~Window();

 private:
  GLFWwindow* window;
  uint32_t width, height;
};
}  // namespace DawnViewer