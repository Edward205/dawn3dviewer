#pragma once

#include "components/mesh.hpp"
#include "entt.hpp"

namespace DawnViewer {
    class Scene {
    public:
        Scene();
        ~Scene();
        entt::entity createEntity();
        void destroyEntity(entt::entity entity);
        std::vector<entt::entity> getRenderables();
        
        template<typename Component, typename... Args>
        void addComponent(entt::entity target, Args&&... args) {
            registry_.emplace<Component>(target, std::forward<Args>(args)...);
        }
        
        template<typename Component>
        void removeComponent(entt::entity entity) {
            registry_.remove<Component>(entity);
        }

        template<typename Component>
        Component* getComponent(entt::entity entity) {
            return registry_.try_get<Component>(entity);
        }
    private:
        entt::registry registry_;
    };
}