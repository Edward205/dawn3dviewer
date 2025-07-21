#pragma once

#include "dawn/webgpu_cpp.h"

struct MeshComponent {
    wgpu::Buffer vertexBuffer;
    uint32_t indexCount;
};