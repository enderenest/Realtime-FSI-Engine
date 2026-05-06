#include "interaction/Picking.h"
#include <glm/gtc/matrix_transform.hpp>
#include <limits>

Ray screenToRay(float mouseX, float mouseY,
                int screenWidth, int screenHeight,
                const glm::mat4& view,
                const glm::mat4& projection)
{
    // Convert screen coords to normalized device coordinates [-1, 1]
    float ndcX = (2.0f * mouseX) / screenWidth - 1.0f;
    float ndcY = 1.0f - (2.0f * mouseY) / screenHeight;

    // Unproject near and far points
    glm::mat4 invVP = glm::inverse(projection * view);

    glm::vec4 nearPoint = invVP * glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
    glm::vec4 farPoint  = invVP * glm::vec4(ndcX, ndcY,  1.0f, 1.0f);

    nearPoint /= nearPoint.w;
    farPoint  /= farPoint.w;

    Ray ray;
    ray.origin    = glm::vec3(nearPoint);
    ray.direction = glm::normalize(glm::vec3(farPoint - nearPoint));
    return ray;
}

// Möller–Trumbore ray-triangle intersection. Returns t >= 0 on hit, -1 on miss.
static float rayTriangleIntersect(const Ray& ray,
                                  const glm::vec3& v0,
                                  const glm::vec3& v1,
                                  const glm::vec3& v2)
{
    const float EPSILON = 1e-7f;
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(ray.direction, edge2);
    float a = glm::dot(edge1, h);

    if (a > -EPSILON && a < EPSILON) return -1.0f; // Parallel

    float f = 1.0f / a;
    glm::vec3 s = ray.origin - v0;
    float u = f * glm::dot(s, h);
    if (u < 0.0f || u > 1.0f) return -1.0f;

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f) return -1.0f;

    float t = f * glm::dot(edge2, q);
    return (t > EPSILON) ? t : -1.0f;
}

PickResult pickVertex(const Ray& ray, const MyMesh& mesh, float /*threshold*/)
{
    PickResult result;
    float closestT = std::numeric_limits<float>::max();
    glm::vec3 hitPoint;
    OpenMesh::FaceHandle hitFace;

    // Find the closest front-facing triangle hit
    for (auto fIt = mesh.faces_begin(); fIt != mesh.faces_end(); ++fIt) {
        // Collect face vertices
        std::vector<glm::vec3> fv;
        for (auto fvIt = mesh.cfv_iter(*fIt); fvIt.is_valid(); ++fvIt) {
            auto p = mesh.point(*fvIt);
            fv.push_back(glm::vec3(p[0], p[1], p[2]));
        }

        // Triangulate polygon fan (works for triangles and quads)
        for (size_t i = 1; i + 1 < fv.size(); i++) {
            float t = rayTriangleIntersect(ray, fv[0], fv[i], fv[i + 1]);
            if (t >= 0.0f && t < closestT) {
                closestT = t;
                hitPoint = ray.origin + t * ray.direction;
                hitFace = *fIt;
                result.hit = true;
            }
        }
    }

    if (!result.hit) return result;

    // Pick the vertex on the hit face closest to the intersection point
    float minDist = std::numeric_limits<float>::max();
    for (auto fvIt = mesh.cfv_iter(hitFace); fvIt.is_valid(); ++fvIt) {
        auto p = mesh.point(*fvIt);
        glm::vec3 v(p[0], p[1], p[2]);
        float d = glm::length(v - hitPoint);
        if (d < minDist) {
            minDist = d;
            result.vertexIndex = fvIt->idx();
            result.distance = closestT;
        }
    }

    return result;
}
