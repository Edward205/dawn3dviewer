#include "core/window.hpp"

#include "GLFW/glfw3.h"
#include "webgpu/webgpu_glfw.h"

namespace DawnViewer {
void Window::create(const char* title, int width, int height) {
  if (!glfwInit()) {
    return;
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window = glfwCreateWindow(width, height, title, nullptr, nullptr);
};
void Window::shutdown() {
  glfwDestroyWindow(window);
  // glfwTerminate();
}
wgpu::Surface Window::createSurface(wgpu::Instance instance) {
    if (!window || !instance) {
        return nullptr;
    }
    return wgpu::glfw::CreateSurfaceForWindow(instance, window);}

Window::~Window() { shutdown(); }
}  // namespace DawnViewer