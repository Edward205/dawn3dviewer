#pragma once

#include "dawn/webgpu_cpp.h"

namespace DawnViewer
{
    struct MeshComponent {
        MeshComponent(std::vector<float>& pointData, wgpu::Device device);
        ~MeshComponent();
        wgpu::Buffer vertexBuffer;
    };
}