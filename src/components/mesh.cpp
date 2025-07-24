#include "components/mesh.hpp"
#include <iostream>

namespace DawnViewer {
    MeshComponent::MeshComponent(std::vector<float>& pointData, wgpu::Device device)
    {
        wgpu::BufferDescriptor bufferDesc{.usage = wgpu::BufferUsage::CopyDst |
                                            wgpu::BufferUsage::Vertex,
                                .size = pointData.size() * sizeof(float),
                                .mappedAtCreation = false};

        vertexBuffer = device.CreateBuffer(&bufferDesc);
        device.GetQueue().WriteBuffer(vertexBuffer, 0, pointData.data(),
                                        bufferDesc.size);
    }
    MeshComponent::~MeshComponent()
    {
        vertexBuffer.Destroy();
    }
}