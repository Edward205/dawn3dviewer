#include "core/renderer.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "GLFW/glfw3.h"
#include "components/mesh.hpp"
#include "components/transform.hpp"
#include "glm/ext/matrix_float4x4.hpp"
#include "resources/mesh_loader.hpp"
namespace DawnViewer {
void Renderer::initDevice(wgpu::Instance instance) {
  // Get adapter
  wgpu::Future f1 = instance.RequestAdapter(
      nullptr, wgpu::CallbackMode::WaitAnyOnly,
      [this](wgpu::RequestAdapterStatus status, wgpu::Adapter a, wgpu::StringView message) {
        if (status != wgpu::RequestAdapterStatus::Success) {
          std::cout << "RequestAdapter: " << std::string(message) << "\n";
          exit(0);  // TODO better error handling
        }
        adapter = std::move(a);
      });
  // TODO: Don't wait infinetly long for the adapter
  instance.WaitAny(f1, UINT64_MAX);

  wgpu::DeviceDescriptor desc{};
  desc.SetUncapturedErrorCallback(
      [](const wgpu::Device &, wgpu::ErrorType errorType, wgpu::StringView message) {
        std::cout << "Error: " << static_cast<int>(errorType)
                  << " - message: " << std::string(message) << "\n";
      });

  // Set device requirements
  wgpu::Limits requiredLimits;
  requiredLimits.maxVertexAttributes = 8;
  requiredLimits.maxVertexBuffers = 4;
  requiredLimits.maxBufferSize = 10000 * sizeof(float) * 9;
  requiredLimits.maxVertexBufferArrayStride = 5 * sizeof(float);
  requiredLimits.maxTextureDimension1D = WGPU_LIMIT_U32_UNDEFINED;
  requiredLimits.maxTextureDimension2D = WGPU_LIMIT_U32_UNDEFINED;
  requiredLimits.maxTextureDimension3D = WGPU_LIMIT_U32_UNDEFINED;
  requiredLimits.maxInterStageShaderVariables = 3;
  requiredLimits.maxVertexBufferArrayStride = 6 * sizeof(float);

  desc.requiredLimits = &requiredLimits;

  // Get device
  wgpu::Future f2 = adapter.RequestDevice(
      &desc, wgpu::CallbackMode::WaitAnyOnly,
      [this](wgpu::RequestDeviceStatus status, wgpu::Device d, wgpu::StringView message) {
        if (status != wgpu::RequestDeviceStatus::Success) {
          std::cout << "RequestDevice: " << std::string(message) << "\n";
          exit(0);  // TODO better error handling
        }
        device = std::move(d);
      });
  // TODO: Don't wait infinetly long for the device
  instance.WaitAny(f2, UINT64_MAX);
}
void Renderer::initSurface(wgpu::Surface surface) {
  // Get the GPU's supported colour space format
  wgpu::SurfaceCapabilities capabilities;
  surface.GetCapabilities(adapter, &capabilities);
  format = capabilities.formats[0];
  wgpu::SurfaceConfiguration config{.device = device,
                                    .format = format,
                                    .width = width,
                                    .height = height,
                                    .presentMode = wgpu::PresentMode::Fifo};
  surface.Configure(&config);
}
void Renderer::initScene() {
  /*entity1 = scene.createEntity();
  std::vector<float> *pointData = new std::vector<float>;
  DawnViewer::loadMeshFromObj("res/mammoth.obj", *pointData);
  scene.addComponent<DawnViewer::MeshComponent>(entity1, *pointData, device);
  scene.addComponent<DawnViewer::TransformComponent>(entity1, glm::vec3(0, 0, 0), glm::vec3(0, 0, 0),
                                                glm::vec3(1));
*/
  entity2 = scene.createEntity();
  std::vector<float> *pointData1 = new std::vector<float>;
  DawnViewer::loadMeshFromObj("res/teapot.obj", *pointData1);
  scene.addComponent<DawnViewer::MeshComponent>(entity2, *pointData1, device);
  scene.addComponent<DawnViewer::TransformComponent>(entity2, glm::vec3(0, 0, 0), glm::vec3(0, 0, 0),
                                                glm::vec3(1));
}
void Renderer::initUniforms() {
  // scene uniforms buffer
  wgpu::BufferDescriptor bufferDesc;
  bufferDesc = {.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
                .size = sizeof(SceneUniforms),
                .mappedAtCreation = false};
  sceneUniformsBuffer = device.CreateBuffer(&bufferDesc);
  SceneUniforms sceneUniforms{
      .view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -5.0f)),
      .projection = glm::perspective(glm::radians(45.0f), (float)width / height, 0.1f, 100.0f),
      .lightDirection = glm::vec3(0, 0, 1),
      .time = 0.0f,
  };
  device.GetQueue().WriteBuffer(sceneUniformsBuffer, 0, &sceneUniforms, sizeof(SceneUniforms));

  // object uniform buffer
  bufferDesc = {.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
                .size = MAX_ENTITIES * DYNAMIC_BUFFER_ALIGNMENT,
                .mappedAtCreation = false};
  objectUniformsBuffer = device.CreateBuffer(&bufferDesc);
  ObjectUniforms objectUniforms{
      .model = glm::rotate(glm::mat4(1.0f), glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
  };
  device.GetQueue().WriteBuffer(objectUniformsBuffer, 0, &objectUniforms, sizeof(ObjectUniforms));
}
void Renderer::initBindGroups() {
  wgpu::BindGroupLayout sceneBindGroupLayout;
  wgpu::BindGroupLayout objectBindGroupLayout;

  // Scene bind group layout
  wgpu::BindGroupLayoutEntry sceneLayoutEntry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
      .buffer = {.type = wgpu::BufferBindingType::Uniform}};
  wgpu::BindGroupLayoutDescriptor sceneLayoutDesc{.entryCount = 1, .entries = &sceneLayoutEntry};
  sceneBindGroupLayout = device.CreateBindGroupLayout(&sceneLayoutDesc);

  // Object bind group layout
  wgpu::BindGroupLayoutEntry objectLayoutEntry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
      .buffer = {.type = wgpu::BufferBindingType::Uniform,
                 .hasDynamicOffset = true,
                 .minBindingSize = sizeof(ObjectUniforms)}};
  wgpu::BindGroupLayoutDescriptor objectLayoutDesc{.entryCount = 1, .entries = &objectLayoutEntry};
  objectBindGroupLayout = device.CreateBindGroupLayout(&objectLayoutDesc);

  // Bind groups
  // Scene bind group
  wgpu::BindGroupEntry sceneBindGroupEntry{
      .binding = 0, .buffer = sceneUniformsBuffer, .size = sizeof(SceneUniforms)};
  wgpu::BindGroupDescriptor sceneBindGroupDesc{
      .layout = sceneBindGroupLayout, .entryCount = 1, .entries = &sceneBindGroupEntry};
  sceneBindGroup = device.CreateBindGroup(&sceneBindGroupDesc);

  // Object bind group
  wgpu::BindGroupEntry objectBindGroupEntry{
      .binding = 0, .buffer = objectUniformsBuffer, .size = sizeof(ObjectUniforms)};
  wgpu::BindGroupDescriptor objectBindGroupDesc{
      .layout = objectBindGroupLayout, .entryCount = 1, .entries = &objectBindGroupEntry};
  objectBindGroup = device.CreateBindGroup(&objectBindGroupDesc);

  // Pipeline layout
  std::vector<wgpu::BindGroupLayout> bindGroupLayouts = {sceneBindGroupLayout,
                                                         objectBindGroupLayout};
  wgpu::PipelineLayoutDescriptor layoutDesc{.bindGroupLayoutCount = bindGroupLayouts.size(),
                                            .bindGroupLayouts = bindGroupLayouts.data()};
  pipelineLayout = device.CreatePipelineLayout(&layoutDesc);
}
void Renderer::initDepthBuffer() {
  wgpu::TextureFormat depthFormat = wgpu::TextureFormat::Depth24Plus;

  wgpu::TextureDescriptor depthTextureDesc{
      .usage = wgpu::TextureUsage::RenderAttachment,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {width, height, 1},
      .format = depthFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
      .viewFormatCount = 1,
      .viewFormats = &depthFormat,
  };
  depthTexture = device.CreateTexture(&depthTextureDesc);

  wgpu::TextureViewDescriptor depthViewDesc{
      .format = depthFormat,
      .dimension = wgpu::TextureViewDimension::e2D,
      .aspect = wgpu::TextureAspect::DepthOnly,
  };
  depthTextureView = depthTexture.CreateView(&depthViewDesc);
}
void Renderer::initPipeline() {
  const char shaderCode[] = R"(
    struct SceneUniforms {
      view: mat4x4<f32>,
      projection: mat4x4<f32>,
      lightDirection: vec3f,
      time: f32,
    }
    struct ObjectUniforms {
      model: mat4x4<f32>,
    }
    @group(0) @binding(0) var<uniform> scene: SceneUniforms;
    @group(1) @binding(0) var<uniform> object: ObjectUniforms;
    struct VertexInput {
        @location(0) position: vec3f,
        @location(1) normal: vec3f,
        @location(2) color: vec3f,
    };
    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) color: vec3f,
        @location(1) normal: vec3f,
    };

    @vertex fn vs_main(in: VertexInput) -> VertexOutput {
        var out: VertexOutput;
        let ratio = 512.0 / 512.0;
        var offset = vec2f(0.0);

        let angle = scene.time;

        let alpha = cos(angle);
        let beta = sin(angle);
        var position = vec3f(
          in.position.x,
          alpha * in.position.y + beta * in.position.z,
          alpha * in.position.z - beta * in.position.y,
        );

        out.position = scene.projection * scene.view * object.model * vec4f(in.position, 1.0);

        out.color = in.color;
        out.normal = (object.model * vec4f(in.normal, 0.0)).xyz;

        return out;
    }
    @fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
        let normal = normalize(in.normal);
        let shading = dot(scene.lightDirection, in.normal);
        let color = in.color * shading;
        return vec4f(color, 1.0);
    }
)";

  wgpu::ShaderSourceWGSL wgsl{{.nextInChain = nullptr, .code = shaderCode}};
  wgpu::ShaderModuleDescriptor shaderModuleDesc{.nextInChain = &wgsl};
  wgpu::ShaderModule shaderModule =
      device.CreateShaderModule(&shaderModuleDesc);

  // Vertex Buffer Layout
  std::vector<wgpu::VertexAttribute> vertexAttribs(3);
  vertexAttribs[0] = {.format = wgpu::VertexFormat::Float32x3, .offset = 0, .shaderLocation = 0};
  vertexAttribs[1] = {.format = wgpu::VertexFormat::Float32x3, .offset = 3 * sizeof(float), .shaderLocation = 1};
  vertexAttribs[2] = {.format = wgpu::VertexFormat::Float32x3, .offset = 6 * sizeof(float), .shaderLocation = 2};

  wgpu::VertexBufferLayout vertexBufferLayout{
      .arrayStride = 9 * sizeof(float),
      .attributeCount = vertexAttribs.size(),
      .attributes = vertexAttribs.data()
  };

  // Depth Stencil State
  wgpu::DepthStencilState depthStencilState{
      .format = wgpu::TextureFormat::Depth24Plus, // Must match initDepthBuffer
      .depthWriteEnabled = true,
      .depthCompare = wgpu::CompareFunction::Less,
  };
  
  wgpu::ColorTargetState colorTarget{.format = this->format}; // Use class member format
  
  wgpu::FragmentState fragmentState{
      .module = shaderModule,
      .entryPoint = "fs_main",
      .targetCount = 1,
      .targets = &colorTarget,
  };

  wgpu::RenderPipelineDescriptor descriptor{
      .layout = pipelineLayout,
      .vertex = {.module = shaderModule, .entryPoint = "vs_main", .bufferCount = 1, .buffers = &vertexBufferLayout},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .depthStencil = &depthStencilState,
      .fragment = &fragmentState,
  };
  pipeline = device.CreateRenderPipeline(&descriptor);
}

void Renderer::init(wgpu::Instance instance, uint32_t w, uint32_t h) {
  width = w;
  height = h;
  initDevice(instance);
  initSurface(surface);

  initScene();
  initUniforms();
  initBindGroups();
  initDepthBuffer();
  initPipeline();
  
  objectDataBuffer.resize(MAX_ENTITIES * DYNAMIC_BUFFER_ALIGNMENT);
}

void Renderer::render(wgpu::Instance instance) {
  // update uniforms
  float t = static_cast<float>(glfwGetTime());
  device.GetQueue().WriteBuffer(sceneUniformsBuffer, offsetof(SceneUniforms, time), &t,
                                sizeof(float));

  // position, rotation, scale demo
  std::vector<entt::entity> renderables = scene.getRenderables();
  const float rotationSpeed = 0.02f;

  auto view = scene.viewComponents<DawnViewer::TransformComponent>();
  int a = 2;
  for (auto entity : view) {
    auto &transform = view.get<DawnViewer::TransformComponent>(entity);

    glm::vec3 upAxis;
    if (a % 2 == 0)
      upAxis = glm::vec3(1.0f, 0.0f, 1.0f);
    else
      upAxis = glm::vec3(1.0f, 5.0f, 0.0f);
    glm::quat rotationDelta = glm::angleAxis(rotationSpeed, upAxis);

    // Apply the new rotation by multiplying with the existing one
    // The order matters: this applies the rotation in the object's local space
    transform.rotation = transform.rotation * rotationDelta;

    // Normalize to prevent floating point drift over time
    transform.rotation = glm::normalize(transform.rotation);
    ++a;
  }

  // update object uniforms
  for (size_t i = 0; i < renderables.size(); ++i) {
    const auto &transform = scene.getComponent<DawnViewer::TransformComponent>(renderables[i]);

    const glm::mat4 model = glm::translate(glm::mat4(1.0f), transform->position) *
                            glm::mat4_cast(transform->rotation) *
                            glm::scale(glm::mat4(1.0f), transform->scale);

    uint8_t *destination =
        objectDataBuffer.data() + (i * DYNAMIC_BUFFER_ALIGNMENT) + offsetof(ObjectUniforms, model);

    memcpy(destination, &model,
           sizeof(glm::mat4));  // i hope this won't explode later
  }
  device.GetQueue().WriteBuffer(objectUniformsBuffer, 0, objectDataBuffer.data(),
                                renderables.size() * DYNAMIC_BUFFER_ALIGNMENT);

  wgpu::SurfaceTexture surfaceTexture;
  surface.GetCurrentTexture(&surfaceTexture);

  wgpu::RenderPassColorAttachment attachment{.view = surfaceTexture.texture.CreateView(),
                                             .loadOp = wgpu::LoadOp::Clear,
                                             .storeOp = wgpu::StoreOp::Store,
                                             .clearValue = wgpu::Color{0.1, 0.1, 0.1, 1.0}};
  wgpu::RenderPassDepthStencilAttachment depthStencilAttachment = {
      .view = depthTextureView,
      .depthLoadOp = wgpu::LoadOp::Clear,
      .depthStoreOp = wgpu::StoreOp::Store,
      .depthClearValue = 1.0f,
      .depthReadOnly = false,
      .stencilLoadOp = wgpu::LoadOp::Undefined,
      .stencilStoreOp = wgpu::StoreOp::Undefined,
      .stencilClearValue = 0,
      .stencilReadOnly = true,
  };

  wgpu::RenderPassDescriptor renderpass{.colorAttachmentCount = 1,
                                        .colorAttachments = &attachment,
                                        .depthStencilAttachment = &depthStencilAttachment};

  wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
  wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderpass);
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, sceneBindGroup);

  int entityIndex = 0;
  for (entt::entity entity : renderables) {
    uint32_t dynamicOffset = entityIndex * 256;

    pass.SetBindGroup(1, objectBindGroup, 1, &dynamicOffset);
    DawnViewer::MeshComponent *pointBuffer = scene.getComponent<DawnViewer::MeshComponent>(entity);
    pass.SetVertexBuffer(0, pointBuffer->vertexBuffer, 0, pointBuffer->vertexBuffer.GetSize());
    pass.Draw(pointBuffer->vertexBuffer.GetSize() / (sizeof(float) * 9));

    ++entityIndex;
  }
  pass.End();

  wgpu::CommandBuffer commands = encoder.Finish();
  device.GetQueue().Submit(1, &commands);

  surface.Present();
  instance.ProcessEvents();
}
void Renderer::resize(uint32_t newWidth, uint32_t newHeight)
{
  width = newWidth;
  height = newHeight;
  initSurface(surface);
  initDepthBuffer();

  // recalculate SceneUniforms projection
  glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)width / height, 0.1f, 100.0f);
  device.GetQueue().WriteBuffer(sceneUniformsBuffer, offsetof(SceneUniforms, projection), &projection, sizeof(projection));
}
void Renderer::setSurface(wgpu::Surface newSurface) {
  surface = newSurface;
}
void Renderer::shutdown() {
  sceneUniformsBuffer.Destroy();
  objectUniformsBuffer.Destroy();
  depthTexture.Destroy();
}

}  // namespace DawnViewer