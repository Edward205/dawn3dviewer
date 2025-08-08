#include <GLFW/glfw3.h>
#include <dawn/webgpu_cpp_print.h>
#include <webgpu/webgpu_cpp.h>
#include <webgpu/webgpu_glfw.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/mat4x4.hpp>
#include <iostream>
#include <vector>

#include "components/mesh.hpp"
#include "components/transform.hpp"
#include "core/window.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/trigonometric.hpp"
#include "resources/mesh_loader.hpp"
#include "scene.hpp"

const uint32_t kWidth = 512;
const uint32_t kHeight = 512;
const uint32_t DYNAMIC_BUFFER_ALIGNMENT = 256;
const uint32_t MAX_ENTITIES = 1000;

wgpu::Instance instance;

wgpu::Adapter adapter;
wgpu::Device device;

wgpu::Surface surface;
wgpu::TextureFormat format;

wgpu::RenderPipeline pipeline;

DawnViewer::Scene scene;
entt::entity entity1;
entt::entity entity2;

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
struct SceneUniforms {
  glm::mat4 view;
  glm::mat4 projection;
  glm::vec3 lightDirection;
  float time;
};
struct ObjectUniforms {
  glm::mat4 model;
};

void Init() {
  // Get WebGPU instance
  wgpu::InstanceDescriptor instanceDesc{
      .capabilities = {.timedWaitAnyEnable = true}};
  instance = wgpu::CreateInstance(&instanceDesc);

  // Get adapter
  wgpu::Future f1 = instance.RequestAdapter(
      nullptr, wgpu::CallbackMode::WaitAnyOnly,
      [](wgpu::RequestAdapterStatus status, wgpu::Adapter a,
         wgpu::StringView message) {
        if (status != wgpu::RequestAdapterStatus::Success) {
          std::cout << "RequestAdapter: " << message << "\n";
          exit(0);  // TODO better error handling
        }
        adapter = std::move(a);
      });
  // TODO: Don't wait infinetly long for the adapter
  instance.WaitAny(f1, UINT64_MAX);

  wgpu::DeviceDescriptor desc{};
  desc.SetUncapturedErrorCallback([](const wgpu::Device &,
                                     wgpu::ErrorType errorType,
                                     wgpu::StringView message) {
    std::cout << "Error: " << errorType << " - message: " << message << "\n";
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
      [](wgpu::RequestDeviceStatus status, wgpu::Device d,
         wgpu::StringView message) {
        if (status != wgpu::RequestDeviceStatus::Success) {
          std::cout << "RequestDevice: " << message << "\n";
          exit(0);  // TODO better error handling
        }
        device = std::move(d);
      });
  // TODO: Don't wait infinetly long for the device
  instance.WaitAny(f2, UINT64_MAX);
}

void ConfigureSurface() {
  // Get the GPU's supported colour space format
  wgpu::SurfaceCapabilities capabilities;
  surface.GetCapabilities(adapter, &capabilities);
  format = capabilities.formats[0];

  wgpu::SurfaceConfiguration config{.device = device,
                                    .format = format,
                                    .width = kWidth,
                                    .height = kHeight,
                                    .presentMode = wgpu::PresentMode::Fifo};
  surface.Configure(&config);
}

wgpu::BindGroup sceneBindGroup;
wgpu::BindGroup objectBindGroup;
wgpu::Buffer sceneUniformsBuffer;
wgpu::Buffer objectUniformsBuffer;
wgpu::Texture depthTexture;
wgpu::TextureView depthTextureView;
float currentTime = 1.0f;
void CreateRenderPipeline() {
  // load a model
  // create entity
  entity1 = scene.createEntity();
  std::vector<float> *pointData = new std::vector<float>;
  DawnViewer::loadMeshFromObj("res/mammoth.obj", *pointData);
  scene.addComponent<DawnViewer::MeshComponent>(entity1, *pointData, device);
  scene.addComponent<DawnViewer::TransformComponent>(
      entity1, glm::vec3(0, 0, 0), glm::vec3(0, 0, 0), glm::vec3(1));

  entity2 = scene.createEntity();
  std::vector<float> *pointData1 = new std::vector<float>;
  DawnViewer::loadMeshFromObj("res/teapot.obj", *pointData1);
  scene.addComponent<DawnViewer::MeshComponent>(entity2, *pointData1, device);
  scene.addComponent<DawnViewer::TransformComponent>(
      entity2, glm::vec3(1, 0, 0), glm::vec3(0, 0, 0), glm::vec3(1));

  // buffer layout
  std::vector<wgpu::VertexAttribute> vertexAttribs(3);

  vertexAttribs[0] = {.format = wgpu::VertexFormat::Float32x3,
                      .offset = 0,
                      .shaderLocation = 0};
  vertexAttribs[1] = {.format = wgpu::VertexFormat::Float32x3,
                      .offset = 3 * sizeof(float),
                      .shaderLocation = 1};
  vertexAttribs[2] = {.format = wgpu::VertexFormat::Float32x3,
                      .offset = 6 * sizeof(float),
                      .shaderLocation = 2};

  wgpu::VertexBufferLayout vertexBufferLayout = {
      .stepMode = wgpu::VertexStepMode::Vertex,
      .arrayStride = 9 * sizeof(float),
      .attributeCount = vertexAttribs.size(),
      .attributes = vertexAttribs.data()};

  // PIPELINE LAYOUT
  wgpu::PipelineLayout layout;

  // uniform buffer
  // scene uniforms buffer
  wgpu::BufferDescriptor bufferDesc;
  bufferDesc = {
      .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
      .size = sizeof(SceneUniforms),
      .mappedAtCreation = false};
  sceneUniformsBuffer = device.CreateBuffer(&bufferDesc);

  SceneUniforms uniforms{
      .view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -5.0f)),
      .projection = glm::perspective(glm::radians(45.0f),
                                     (float)kWidth / kHeight, 0.1f, 100.0f),
      .lightDirection = glm::vec3(0, 0, 1),
      .time = 0.0f,
  };
  device.GetQueue().WriteBuffer(sceneUniformsBuffer, 0, &uniforms,
                                sizeof(SceneUniforms));

  // object uniform buffer
  bufferDesc = {
      .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
      .size = MAX_ENTITIES * DYNAMIC_BUFFER_ALIGNMENT,
      .mappedAtCreation = false};
  objectUniformsBuffer = device.CreateBuffer(&bufferDesc);

  ObjectUniforms objectUniforms{
      .model = glm::rotate(glm::mat4(1.0f), glm::radians(-55.0f),
                           glm::vec3(1.0f, 0.0f, 0.0f)),
  };
  device.GetQueue().WriteBuffer(objectUniformsBuffer, 0, &objectUniforms,
                                sizeof(ObjectUniforms));

  // binding for the scene uniforms buffer
  wgpu::BindGroupEntry sceneBindGroupEntry{.nextInChain = nullptr,
                                           .binding = 0,
                                           .buffer = sceneUniformsBuffer,
                                           .size = sizeof(SceneUniforms)};

  // binding for the scene uniforms buffer
  wgpu::BindGroupEntry objectBindGroupEntry{.nextInChain = nullptr,
                                            .binding = 0,
                                            .buffer = objectUniformsBuffer,
                                            .size = sizeof(ObjectUniforms)};

  // binding layout for the object binding
  wgpu::BindGroupLayoutEntry sceneLayoutEntry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
      .buffer = {
          .type = wgpu::BufferBindingType::Uniform,
          .minBindingSize = sizeof(SceneUniforms),
      }};
  // binding layout for the object binding
  wgpu::BindGroupLayoutEntry objectLayoutEntry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment,
      .buffer = {
          .type = wgpu::BufferBindingType::Uniform,
          .hasDynamicOffset = true,
          .minBindingSize = sizeof(ObjectUniforms),
      }};

  // scene bind group layout
  wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc{
      .nextInChain = nullptr,
      .entryCount = 1,
      .entries = &sceneLayoutEntry,
  };
  wgpu::BindGroupLayout sceneBindGroupLayout;
  sceneBindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc);

  // scene bind group layout
  bindGroupLayoutDesc = {
      .nextInChain = nullptr,
      .entryCount = 1,
      .entries = &objectLayoutEntry,
  };
  wgpu::BindGroupLayout objectBindGroupLayout;
  objectBindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc);

  // scene bind group
  wgpu::BindGroupDescriptor sceneBindGroupDescriptor{
      .nextInChain = nullptr,
      .layout = sceneBindGroupLayout,
      .entryCount = 1,
      .entries = &sceneBindGroupEntry,
  };
  sceneBindGroup = device.CreateBindGroup(&sceneBindGroupDescriptor);

  // object bind group
  wgpu::BindGroupDescriptor objectBindGroupDescriptor{
      .nextInChain = nullptr,
      .layout = objectBindGroupLayout,
      .entryCount = 1,
      .entries = &objectBindGroupEntry,
  };
  objectBindGroup = device.CreateBindGroup(&objectBindGroupDescriptor);

  // layout
  std::vector<wgpu::BindGroupLayout> bindGroupLayouts = {sceneBindGroupLayout,
                                                         objectBindGroupLayout};
  wgpu::PipelineLayoutDescriptor layoutDesc{
      .nextInChain = nullptr,
      .bindGroupLayoutCount = bindGroupLayouts.size(),
      .bindGroupLayouts = bindGroupLayouts.data(),
  };
  layout = device.CreatePipelineLayout(&layoutDesc);

  // END PIPELINE LAYOUT

  // z buffer
  // configure depth stencil
  wgpu::TextureFormat depthTextureFormat = wgpu::TextureFormat::Depth24Plus;
  wgpu::DepthStencilState depthStencilState{
      .format = depthTextureFormat,
      .depthWriteEnabled = true,
      .depthCompare = wgpu::CompareFunction::Less,
      .stencilReadMask = 0,
      .stencilWriteMask = 0,
  };

  // depth texture
  wgpu::TextureDescriptor depthTextureDesc{
      .usage = wgpu::TextureUsage::RenderAttachment,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {512, 512, 1},
      .format = depthTextureFormat,
      .mipLevelCount = 1,
      .sampleCount = 1,
      .viewFormatCount = 1,
      .viewFormats = &depthTextureFormat,
  };
  depthTexture = device.CreateTexture(&depthTextureDesc);
  // Create the view of the depth texture manipulated by the rasterizer
  wgpu::TextureViewDescriptor depthTextureViewDesc{
      .format = depthTextureFormat,
      .dimension = wgpu::TextureViewDimension::e2D,
      .baseMipLevel = 0,
      .mipLevelCount = 1,
      .baseArrayLayer = 0,
      .arrayLayerCount = 1,
      .aspect = wgpu::TextureAspect::DepthOnly,
  };
  depthTextureView = depthTexture.CreateView(&depthTextureViewDesc);

  wgpu::ShaderSourceWGSL wgsl{{.nextInChain = nullptr, .code = shaderCode}};

  wgpu::ShaderModuleDescriptor shaderModuleDescriptor{.nextInChain = &wgsl};
  wgpu::ShaderModule shaderModule =
      device.CreateShaderModule(&shaderModuleDescriptor);

  wgpu::ColorTargetState colorTargetState{.format = format};

  wgpu::FragmentState fragmentState{
      .module = shaderModule,
      .entryPoint = "fs_main",
      .targetCount = 1,
      .targets = &colorTargetState,
  };

  wgpu::RenderPipelineDescriptor descriptor{
      .layout = layout,
      .vertex =
          {
              .module = shaderModule,
              .entryPoint = "vs_main",
              .bufferCount = 1,
              .buffers = &vertexBufferLayout,
          },
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .depthStencil = &depthStencilState,
      .fragment = &fragmentState,
  };
  pipeline = device.CreateRenderPipeline(&descriptor);
}

void InitGraphics() {
  ConfigureSurface();
  CreateRenderPipeline();
}

std::vector<uint8_t> objectDataBuffer(MAX_ENTITIES *DYNAMIC_BUFFER_ALIGNMENT);
void Render() {
  // update uniforms
  float t = static_cast<float>(glfwGetTime());
  device.GetQueue().WriteBuffer(
      sceneUniformsBuffer, offsetof(SceneUniforms, time), &t, sizeof(float));

  // position, rotation, scale demo
  std::vector<entt::entity> renderables = scene.getRenderables();
  const float rotationSpeed = 0.05f;

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
    const auto &transform =
        scene.getComponent<DawnViewer::TransformComponent>(renderables[i]);

    const glm::mat4 model =
        glm::translate(glm::mat4(1.0f), transform->position) *
        glm::mat4_cast(transform->rotation) *
        glm::scale(glm::mat4(1.0f), transform->scale);

    uint8_t *destination = objectDataBuffer.data() +
                           (i * DYNAMIC_BUFFER_ALIGNMENT) +
                           offsetof(ObjectUniforms, model);

    memcpy(destination, &model,
           sizeof(glm::mat4));  // i hope this won't explode later
  }
  device.GetQueue().WriteBuffer(objectUniformsBuffer, 0,
                                objectDataBuffer.data(),
                                renderables.size() * DYNAMIC_BUFFER_ALIGNMENT);

  wgpu::SurfaceTexture surfaceTexture;
  surface.GetCurrentTexture(&surfaceTexture);

  wgpu::RenderPassColorAttachment attachment{
      .view = surfaceTexture.texture.CreateView(),
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

  wgpu::RenderPassDescriptor renderpass{
      .colorAttachmentCount = 1,
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
    DawnViewer::MeshComponent *pointBuffer =
        scene.getComponent<DawnViewer::MeshComponent>(entity);
    pass.SetVertexBuffer(0, pointBuffer->vertexBuffer, 0,
                         pointBuffer->vertexBuffer.GetSize());
    pass.Draw(pointBuffer->vertexBuffer.GetSize() / (sizeof(float) * 9));

    ++entityIndex;
  }
  pass.End();

  wgpu::CommandBuffer commands = encoder.Finish();
  device.GetQueue().Submit(1, &commands);
}

void Start() {
  if (!glfwInit()) {
    return;
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow *window =
      glfwCreateWindow(kWidth, kHeight, "WebGPU window", nullptr, nullptr);

  surface = wgpu::glfw::CreateSurfaceForWindow(instance, window);

  InitGraphics();

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    Render();
    surface.Present();
    instance.ProcessEvents();
  }
}

int main() {
  DawnViewer::Window window;
  window.create("test", 50, 50);
  Init();
  Start();
  // TODO: destroy
}
