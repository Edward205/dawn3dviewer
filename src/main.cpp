

#include "core/renderer.hpp"
#include "core/window.hpp"
#include "spdlog/spdlog.h"

#define INITIAL_WIDTH 1280
#define INITIAL_HEIGHT 720

int main() {
  spdlog::set_level(spdlog::level::debug);
  spdlog::info("Project DawnViewer");

  DawnViewer::Window window;
  DawnViewer::Renderer renderer;
  window.create("test", INITIAL_WIDTH, INITIAL_HEIGHT);

  wgpu::Instance instance;
  wgpu::InstanceDescriptor instanceDesc{.capabilities = {.timedWaitAnyEnable = true}};
  instance = wgpu::CreateInstance(&instanceDesc);

  window.createSurface(instance);
  renderer.setSurface(window.getSurface());
  
  window.onResize([&renderer](int w, int h) {
    renderer.resize(w, h);
  });

  renderer.init(instance, INITIAL_WIDTH, INITIAL_HEIGHT);

  while (true) {
    glfwPollEvents();
    renderer.render(instance);
  }

  renderer.shutdown();
  window.shutdown();
  glfwTerminate();
}
