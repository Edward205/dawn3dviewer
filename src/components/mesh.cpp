#include "components/mesh.hpp"
#include <iostream>

namespace DawnViewer {
    MeshComponent::MeshComponent(std::vector<float>& pointData, wgpu::Device device)
    {
        // TODO: don't use logic in the component. make a separate asset manager, keep only a reference to the asset.
        // declare a path or a unique ID (hash) variable to point to the asset
        
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