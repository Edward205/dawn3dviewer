

#include "core/renderer.hpp"
#include "core/window.hpp"
int main() {
  DawnViewer::Window window;
  DawnViewer::Renderer renderer;

  window.create("test", 512, 512);

  wgpu::Instance instance;
  wgpu::InstanceDescriptor instanceDesc{.capabilities = {.timedWaitAnyEnable = true}};
  instance = wgpu::CreateInstance(&instanceDesc);

  window.createSurface(instance);
  renderer.setSurface(window.getSurface());
  
  window.onResize([&renderer](int w, int h) {
    renderer.resize(w, h);
  });

  renderer.init(instance, 512, 512);

  while (true) {
    glfwPollEvents();
    renderer.render(instance);
  }

  renderer.shutdown();
  window.shutdown();
  glfwTerminate();
}
