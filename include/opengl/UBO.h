#ifndef UBO_CLASS_H
#define UBO_CLASS_H

#include <cstddef>
#include <glad/glad.h>
#include "core/Types.h"

template<typename T>
class UBO {
public:
    // Construct a UBO to hold a single struct of type T
    UBO(GLenum usage = GL_DYNAMIC_DRAW);

    // Destructor: deletes the GPU buffer
    ~UBO();

    // Upload the struct data to the GPU buffer
    void upload(const T& data);

    // Bind this UBO to the given binding point in GLSL
    void bindTo(GLuint bindingIndex) const;

    GLuint getID() const;

private:
    GLuint _id;       // OpenGL buffer handle
};

#endif // UBO_CLASS_H