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

  // set current window as active
  glfwSetWindowUserPointer(window, this);

  glfwSetFramebufferSizeCallback(window, [](GLFWwindow* glfwWin, int w, int h) {
    auto* windowInstance = static_cast<Window*>(glfwGetWindowUserPointer(glfwWin));

    if (windowInstance && windowInstance->resizeCallback) {
      windowInstance->resizeCallback(w, h);
    }
  });
};
void Window::onResize(std::function<void(int width, int height)> callback) {
  resizeCallback = callback;
}

void Window::shutdown() { glfwDestroyWindow(window); }
wgpu::Surface Window::createSurface(wgpu::Instance instance) {
  return wgpu::glfw::CreateSurfaceForWindow(instance, window);
}
bool Window::shouldClose() { return glfwWindowShouldClose(window); }

}  // namespace DawnViewer