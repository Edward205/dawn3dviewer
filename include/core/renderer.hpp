#pragma once

#include "components/mesh.hpp"
#include "glm/ext/matrix_float4x4.hpp"
#include "scene.hpp"

namespace DawnViewer {
class Renderer {
 public:
  void init(wgpu::Instance instance, uint32_t w, uint32_t h);
  void shutdown();
  void render(wgpu::Instance instance);
  void resize(uint32_t newWidth, uint32_t newHeight);
  void setSurface(wgpu::Surface newSurface);
 private:
  wgpu::Adapter adapter;
  wgpu::Device device;
  wgpu::Surface surface;

  wgpu::Buffer sceneUniformsBuffer;
  wgpu::Buffer objectUniformsBuffer;
  wgpu::BindGroup sceneBindGroup;
  wgpu::BindGroup objectBindGroup;
  wgpu::Texture depthTexture;
  wgpu::TextureView depthTextureView;
  wgpu::PipelineLayout pipelineLayout;
  wgpu::RenderPipeline pipeline;

  uint32_t width = 512;
  uint32_t height = 512;
  const uint32_t DYNAMIC_BUFFER_ALIGNMENT = 256;
  const uint32_t MAX_ENTITIES = 1000;

  wgpu::TextureFormat format;

  DawnViewer::Scene scene;
  entt::entity entity1;
  entt::entity entity2;

  struct SceneUniforms {
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec3 lightDirection;
    float time;
  };
  struct ObjectUniforms {
    glm::mat4 model;
  };
  std::vector<uint8_t> objectDataBuffer;

  void initDevice(wgpu::Instance instance);
  void initSurface(wgpu::Surface surface);
  void initScene(); // TODO: Scene* parameter
  void initUniforms();
  void initBindGroups();
  void initDepthBuffer();
  void initPipeline();
};
}  // namespace DawnViewer