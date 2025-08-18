#include "core/renderer.hpp"
#include "core/window.hpp"
#include "spdlog/spdlog.h"

#define INITIAL_WIDTH 1280
#define INITIAL_HEIGHT 720

int main() {
  spdlog::set_level(spdlog::level::debug);
  spdlog::info("Project DawnViewer");

  DawnViewer::Renderer renderer;
  DawnViewer::Window window;
  window.create("test", INITIAL_WIDTH, INITIAL_HEIGHT);

  wgpu::Instance instance;
  wgpu::InstanceDescriptor instanceDesc{.capabilities = {.timedWaitAnyEnable = true}};
  instance = wgpu::CreateInstance(&instanceDesc);

  wgpu::Surface surface = window.createSurface(instance);
  renderer.setSurface(std::move(surface));

  window.onResize([&renderer](int w, int h) { renderer.resize(w, h); });

  renderer.init(instance, INITIAL_WIDTH, INITIAL_HEIGHT);

  while (!window.shouldClose()) {
    glfwPollEvents();
    renderer.render(instance);
  }

  renderer.shutdown();
  window.shutdown();
  glfwTerminate();
}
