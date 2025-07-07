#include <cmath>
#include <iostream>

#include <GLFW/glfw3.h>
#include <dawn/webgpu_cpp_print.h>
#include <webgpu/webgpu_cpp.h>
#include <webgpu/webgpu_glfw.h>

#include "FileLoader.h"

const uint32_t kWidth = 512;
const uint32_t kHeight = 512;

wgpu::Instance instance;

wgpu::Adapter adapter;
wgpu::Device device;

wgpu::Surface surface;
wgpu::TextureFormat format;

wgpu::RenderPipeline pipeline;

const char shaderCode[] = R"(
    @group(0) @binding(0) var<uniform> uTime: f32;

    struct VertexInput {
        @location(0) position: vec2f,
        @location(1) color: vec3f,
    };
    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) color: vec3f,
    };

    @vertex fn vs_main(in: VertexInput) -> VertexOutput {
        let ratio = 512.0 / 512.0;
        let offset = vec2f(-0.6875, -0.463);

        offset += 0.3 * vec2f(cos(uTime), sin(uTime));
        
        var out: VertexOutput;
        out.position = vec4f(in.position.x + offset.x, (in.position.y + offset.y) * ratio, 0.0, 1.0);
        out.color = in.color;
        return out;
    }
    @fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
        return vec4f(in.color, 1.0);
    }
)";

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
          exit(0); // TODO better error handling
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
  requiredLimits.maxBufferSize = 128 * 5 * sizeof(float);
  requiredLimits.maxVertexBufferArrayStride = 5 * sizeof(float);
  requiredLimits.maxTextureDimension1D = WGPU_LIMIT_U32_UNDEFINED;
  requiredLimits.maxTextureDimension2D = WGPU_LIMIT_U32_UNDEFINED;
  requiredLimits.maxTextureDimension3D = WGPU_LIMIT_U32_UNDEFINED;
  requiredLimits.maxInterStageShaderVariables = 3;

  requiredLimits.maxBindGroups = 1;
  requiredLimits.maxUniformBuffersPerShaderStage = 1;
  requiredLimits.maxUniformBufferBindingSize = 16 * 4;

  desc.requiredLimits = &requiredLimits;

  // Get device
  wgpu::Future f2 = adapter.RequestDevice(
      &desc, wgpu::CallbackMode::WaitAnyOnly,
      [](wgpu::RequestDeviceStatus status, wgpu::Device d,
         wgpu::StringView message) {
        if (status != wgpu::RequestDeviceStatus::Success) {
          std::cout << "RequestDevice: " << message << "\n";
          exit(0); // TODO better error handling
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

wgpu::Buffer pointBuffer;
wgpu::Buffer indexBuffer;
wgpu::Buffer uniformBuffer;
wgpu::BindGroup bindGroup;
uint32_t indexCount;
float currentTime = 1.0f;
void CreateRenderPipeline() {
  // load a model

  std::vector<float> pointData;
  std::vector<uint16_t> indexData;

  bool success = loadGeometry("res/webgpu.txt", pointData, indexData);

  if (!success) {
    std::cerr << "Could not load geometry!" << std::endl;
    exit(1);
  }

  indexCount = static_cast<uint32_t>(indexData.size());

  // create point buffer
  wgpu::BufferDescriptor bufferDesc{.usage = wgpu::BufferUsage::CopyDst |
                                             wgpu::BufferUsage::Vertex,
                                    .size = pointData.size() * sizeof(float),
                                    .mappedAtCreation = false};

  pointBuffer = device.CreateBuffer(&bufferDesc);
  device.GetQueue().WriteBuffer(pointBuffer, 0, pointData.data(),
                                bufferDesc.size);

  // create index buffer
  bufferDesc = {.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Index,
                .size = indexData.size() * sizeof(uint16_t)};
  // uint16_t is 2 bytes, we need to pad to a multiple of 4
  bufferDesc.size = (bufferDesc.size + 3) & ~3;
  indexData.resize((indexData.size() + 1) & ~1);

  indexBuffer = device.CreateBuffer(&bufferDesc);
  device.GetQueue().WriteBuffer(indexBuffer, 0, indexData.data(),
                                bufferDesc.size);

  // create uniform buffer
  // size due to alignment
  bufferDesc = {.usage =
                    wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform,
                .size = 4 * sizeof(float)};
  uniformBuffer = device.CreateBuffer(&bufferDesc);
  device.GetQueue().WriteBuffer(uniformBuffer, 0, &currentTime, sizeof(float));

  // --- pipeline layout---

  wgpu::BindGroupLayoutEntry bindingGroupEntry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Vertex,
      .buffer =
          {
              .type = wgpu::BufferBindingType::Uniform,
              .minBindingSize = 4 * sizeof(float),
          },
  };

  wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc{
      .entryCount = 1,
      .entries = &bindingGroupEntry,
  };
  wgpu::BindGroupLayout bindGroupLayout =
      device.CreateBindGroupLayout(&bindGroupLayoutDesc);

  wgpu::BindGroupEntry binding{
      .binding = 0,
      .buffer = uniformBuffer,
      .offset = 0,
      .size = 4 * sizeof(float),
  };
  wgpu::BindGroupDescriptor bindGroupDesc{
      .layout = bindGroupLayout,
      .entryCount = 1,
      .entries = &binding,
  };
  bindGroup = device.CreateBindGroup(&bindGroupDesc);

  wgpu::PipelineLayoutDescriptor layoutDesc{
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &bindGroupLayout,
  };
  wgpu::PipelineLayout layout = device.CreatePipelineLayout(&layoutDesc);

  // --- end of pipeline layout ---

  // buffer layout
  std::vector<wgpu::VertexAttribute> vertexAttribs(2);

  vertexAttribs[0] = {.format = wgpu::VertexFormat::Float32x2,
                      .offset = 0,
                      .shaderLocation = 0};
  vertexAttribs[1] = {.format = wgpu::VertexFormat::Float32x3,
                      .offset = 2 * sizeof(float),
                      .shaderLocation = 1};

  wgpu::VertexBufferLayout vertexBufferLayout = {
      .stepMode = wgpu::VertexStepMode::Vertex,
      .arrayStride = 5 * sizeof(float),
      .attributeCount = vertexAttribs.size(),
      .attributes = vertexAttribs.data()};

  wgpu::ShaderSourceWGSL wgsl{{.nextInChain = nullptr, .code = shaderCode}};
  
    wgpu::ShaderModuleWGSLDescriptor wgslDesc{};
    wgslDesc.code = shaderCode;

    wgpu::ShaderModuleDescriptor shaderModuleDescriptor{};
    shaderModuleDescriptor.nextInChain = &wgslDesc;

    wgpu::ShaderModule shaderModule =
    device.CreateShaderModule(&shaderModuleDescriptor);

  wgpu::ColorTargetState colorTargetState{.format = format};

  wgpu::FragmentState fragmentState{
      .module = shaderModule, .entryPoint = "fs_main", .targetCount = 1, .targets = &colorTargetState};

  wgpu::RenderPipelineDescriptor descriptor{
      .layout = layout,
      .vertex = {.module = shaderModule,
                 .entryPoint = "vs_main",
                 .bufferCount = 1,
                 .buffers = &vertexBufferLayout},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .fragment = &fragmentState,
  };
  pipeline = device.CreateRenderPipeline(&descriptor);
}

void InitGraphics() {
  ConfigureSurface();
  CreateRenderPipeline();
}

void Render() {
  wgpu::SurfaceTexture surfaceTexture;
  surface.GetCurrentTexture(&surfaceTexture);

  wgpu::RenderPassColorAttachment attachment{
      .view = surfaceTexture.texture.CreateView(),
      .loadOp = wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store};

  wgpu::RenderPassDescriptor renderpass{.colorAttachmentCount = 1,
                                        .colorAttachments = &attachment};

  wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
  wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderpass);
  pass.SetPipeline(pipeline);

  pass.SetVertexBuffer(0, pointBuffer, 0, pointBuffer.GetSize());
  pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint16, 0,
                      indexBuffer.GetSize());
  pass.SetBindGroup(0, bindGroup, 0, nullptr);

  pass.DrawIndexed(indexCount);

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
  Init();
  Start();
  // TODO: destroy
}
