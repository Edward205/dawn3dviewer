#pragma once

#include "entt.hpp"

namespace DawnViewer {
    class Scene {
    public:
        Scene();
        ~Scene();
        entt::entity createEntity();
        void addComponent(entt::entity);
    private:
        entt::registry registry_;
    };
}