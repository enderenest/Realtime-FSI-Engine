#pragma once

#include <glm/glm.hpp>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct PickResult {
    bool hit = false;
    int vertexIndex = -1;
    float distance = 0.0f;
};

// Cast a ray from screen coordinates into world space
Ray screenToRay(float mouseX, float mouseY,
                int screenWidth, int screenHeight,
                const glm::mat4& view,
                const glm::mat4& projection);

// Find the closest vertex hit by the ray on the mesh
PickResult pickVertex(const Ray& ray, const MyMesh& mesh, float threshold = 0.05f);
