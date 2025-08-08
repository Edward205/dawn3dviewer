#include "scene.hpp"

#include "components/mesh.hpp"
#include "entt.hpp"

namespace DawnViewer {
Scene::Scene() {}
Scene::~Scene() {}
entt::entity Scene::createEntity() { return registry_.create(); }
void Scene::destroyEntity(entt::entity entity) { registry_.destroy(entity); }
std::vector<entt::entity> Scene::getRenderables() {
  std::vector<entt::entity> renderables;
  for (auto entity : registry_.view<MeshComponent>())
    renderables.push_back(entity);
  return renderables;
}
}  // namespace DawnViewer