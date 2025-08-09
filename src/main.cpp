

#include "core/renderer.hpp"
#include "core/window.hpp"
int main() {
  DawnViewer::Window window;
  window.create("test", 512, 512);

  wgpu::Instance instance;
  wgpu::InstanceDescriptor instanceDesc{
      .capabilities = {.timedWaitAnyEnable = true}};
  instance = wgpu::CreateInstance(&instanceDesc);

  wgpu::Surface surface = window.createSurface(instance);

  DawnViewer::Renderer renderer;
  renderer.init(instance, surface);

  while(true)
  {
    renderer.render(instance, surface);
  }

  // TODO: destroy
}
