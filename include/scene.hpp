#pragma once

#include "entt.hpp"

namespace DawnViewer {
class Scene {
 public:
  Scene();
  ~Scene();
  entt::entity createEntity();
  void destroyEntity(entt::entity entity);
  std::vector<entt::entity> getRenderables();

  template <typename Component, typename... Args>
  void addComponent(entt::entity target, Args&&... args) {
    registry_.emplace<Component>(target, std::forward<Args>(args)...);
  }

  template <typename Component>
  void removeComponent(entt::entity entity) {
    registry_.remove<Component>(entity);
  }

  template <typename Component>
  Component* getComponent(entt::entity entity) {
    return registry_.try_get<Component>(entity);
  }

  template <typename Component, typename... Func>
  void patchComponent(entt::entity entity, Func&&... func) {
    registry_.patch<Component>(entity, std::forward<Func>(func)...);
  }

  template <typename Component>  // TODO: this might not be aligned with the
                                 // rest of the functions
                                 auto viewComponents() {
    return registry_.view<Component>();
  }

 private:
  entt::registry registry_;
};
}  // namespace DawnViewer