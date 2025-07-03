#include <cmath>
#include <cstdint>
#include <iostream>

#include <GLFW/glfw3.h>
#include <dawn/webgpu_cpp_print.h>
#include <unistd.h>
#include <webgpu/webgpu_cpp.h>
#include <webgpu/webgpu_glfw.h>

const uint32_t kWidth = 512;
const uint32_t kHeight = 512;

wgpu::Instance instance;

wgpu::Adapter adapter;
wgpu::Device device;

wgpu::Surface surface;
wgpu::TextureFormat format;

wgpu::RenderPipeline pipeline;

const char shaderCode[] = R"(
    @vertex fn vertexMain(@location(0) in_vertex_position: vec2f) -> @builtin(position) vec4f {
        return vec4f(in_vertex_position, 0.0, 1.0);
    }
    @fragment fn fragmentMain() -> @location(0) vec4f {
        return vec4f(1, 0, 0, 1);
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
  requiredLimits.maxVertexBuffers = 1;
  requiredLimits.maxBufferSize = 6 * 2 * sizeof(float);
  requiredLimits.maxVertexBufferArrayStride = 2 * sizeof(float);
  requiredLimits.maxTextureDimension1D = WGPU_LIMIT_U32_UNDEFINED;
  requiredLimits.maxTextureDimension2D = WGPU_LIMIT_U32_UNDEFINED;
  requiredLimits.maxTextureDimension3D = WGPU_LIMIT_U32_UNDEFINED;

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

wgpu::Buffer vertexBuffer;
uint32_t vertexCount;
void CreateRenderPipeline() {
  // load a model
  std::vector<float> vertexData = {
    // x0, y0
    -0.5, -0.5,

    // x1, y1
    +0.5, -0.5,

    // x2, y2
    +0.0, +0.5
  };
  vertexCount = static_cast<uint32_t>(vertexData.size() / 2);

  wgpu::BufferDescriptor bufferDesc{.label = "model buffer", .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Vertex, .size = vertexData.size() * sizeof(float), .mappedAtCreation = false};
  vertexBuffer = device.CreateBuffer(&bufferDesc);

  device.GetQueue().WriteBuffer(vertexBuffer, 0, vertexData.data(), bufferDesc.size);

  // buffer layout
  wgpu::VertexAttribute positionAttrib{.format = wgpu::VertexFormat::Float32x2, .offset = 0, .shaderLocation = 0,};

  wgpu::VertexBufferLayout vertexBufferLayout{.stepMode = wgpu::VertexStepMode::Vertex, .arrayStride = 2 * sizeof(float),  .attributeCount = 1, .attributes = &positionAttrib};


  wgpu::ShaderSourceWGSL wgsl{{.nextInChain = nullptr, .code = shaderCode}};

  wgpu::ShaderModuleDescriptor shaderModuleDescriptor{.nextInChain = &wgsl};
  wgpu::ShaderModule shaderModule =
      device.CreateShaderModule(&shaderModuleDescriptor);

  wgpu::ColorTargetState colorTargetState{.format = format};

  wgpu::FragmentState fragmentState{
      .module = shaderModule, .targetCount = 1, .targets = &colorTargetState};

  wgpu::RenderPipelineDescriptor descriptor{
      .vertex = {.module = shaderModule, .bufferCount = 1, .buffers = &vertexBufferLayout},
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
  
  pass.SetVertexBuffer(0, vertexBuffer, 0, vertexBuffer.GetSize());
  pass.Draw(vertexCount, 1, 0, 0);

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
