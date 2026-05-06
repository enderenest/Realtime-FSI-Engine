#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>

class Gizmo {
public:
    Gizmo();
    ~Gizmo();

    void render(GLuint shaderID,
                const glm::mat4& view,
                const glm::mat4& projection,
                const std::vector<glm::vec3>& anchorPositions,
                const glm::vec3* controlPosition);

private:
    GLuint _vao;
    GLuint _vbo;
    float _pointSize = 80.0f;
};
