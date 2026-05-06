#include "interaction/Gizmo.h"
#include <glm/gtc/type_ptr.hpp>

Gizmo::Gizmo() {
    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_vbo);

    glBindVertexArray(_vao);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

Gizmo::~Gizmo() {
    glDeleteBuffers(1, &_vbo);
    glDeleteVertexArrays(1, &_vao);
}

void Gizmo::render(GLuint shaderID,
                   const glm::mat4& view,
                   const glm::mat4& projection,
                   const std::vector<glm::vec3>& anchorPositions,
                   const glm::vec3* controlPosition)
{
    glUseProgram(shaderID);
    glUniformMatrix4fv(glGetUniformLocation(shaderID, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shaderID, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniform1f(glGetUniformLocation(shaderID, "pointSize"), _pointSize);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glBindVertexArray(_vao);

    // Draw anchors in red
    if (!anchorPositions.empty()) {
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        glBufferData(GL_ARRAY_BUFFER,
                     anchorPositions.size() * sizeof(glm::vec3),
                     anchorPositions.data(),
                     GL_DYNAMIC_DRAW);

        glUniform3f(glGetUniformLocation(shaderID, "gizmoColor"), 1.0f, 0.0f, 0.0f);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(anchorPositions.size()));
    }

    // Draw control point in green
    if (controlPosition) {
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3), controlPosition, GL_DYNAMIC_DRAW);

        glUniform3f(glGetUniformLocation(shaderID, "gizmoColor"), 0.0f, 1.0f, 0.0f);
        glDrawArrays(GL_POINTS, 0, 1);
    }

    glBindVertexArray(0);
    glDisable(GL_PROGRAM_POINT_SIZE);
}
