#include "core/renderer.hpp"
#include "core/window.hpp"
#include "spdlog/spdlog.h"

#define INITIAL_WIDTH 1280
#define INITIAL_HEIGHT 720

int main(int argc, char *argv[]) {
  spdlog::set_level(spdlog::level::debug);
  spdlog::info("DawnViewer");
  
  if(argc < 2)
    spdlog::error("Missing model to open. Example: dawn3dviewer teapot.obj");
  spdlog::info("Loading model: {0}", argv[1]);

  DawnViewer::Renderer renderer;
  DawnViewer::Window window;
  window.create("Dawn Viewer", INITIAL_WIDTH, INITIAL_HEIGHT);

  wgpu::Instance instance;
  wgpu::InstanceDescriptor instanceDesc{.capabilities = {.timedWaitAnyEnable = true}};
  instance = wgpu::CreateInstance(&instanceDesc);

  wgpu::Surface surface = window.createSurface(instance);
  renderer.setSurface(std::move(surface));

  window.onResize([&renderer](int w, int h) { renderer.resize(w, h); });

  renderer.init(instance, INITIAL_WIDTH, INITIAL_HEIGHT, argv[1]);

  while (!window.shouldClose()) {
    glfwPollEvents();
    renderer.render(instance);
  }

  renderer.shutdown();
  window.shutdown();
  glfwTerminate();
}
