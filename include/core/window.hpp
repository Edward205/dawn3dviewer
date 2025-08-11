#pragma once

#include "GLFW/glfw3.h"
#include "webgpu/webgpu_cpp.h"

namespace DawnViewer {
class Window {
 public:
  void create(const char* title, int width, int height);
  void createSurface(wgpu::Instance instance);
  wgpu::Surface getSurface();
  void onResize(std::function<void(int width, int height)> callback);
  void shutdown();

 private:
  GLFWwindow* window;
  uint32_t width, height;
  std::function<void(int, int)> resizeCallback;
  wgpu::Surface surface;
};
}  // namespace DawnViewer