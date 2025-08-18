#pragma once

#include "GLFW/glfw3.h"
#include "webgpu/webgpu_cpp.h"

namespace DawnViewer {
class Window {
 public:
  void create(const char* title, int width, int height);
  wgpu::Surface createSurface(wgpu::Instance instance);
  void onResize(std::function<void(int width, int height)> callback);
  bool shouldClose();
  void shutdown();

 private:
  GLFWwindow* window;
  uint32_t width, height;
  std::function<void(int, int)> resizeCallback;
};
}  // namespace DawnViewer