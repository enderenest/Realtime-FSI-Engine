#include "opengl/UBO.h"

template<typename T>
UBO<T>::UBO(GLenum usage)
{
    glGenBuffers(1, &_id);
    glBindBuffer(GL_UNIFORM_BUFFER, _id);
    // Allocate memory for exactly one instance of type T
    glBufferData(GL_UNIFORM_BUFFER, sizeof(T), nullptr, usage);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

template<typename T>
UBO<T>::~UBO() {
    glDeleteBuffers(1, &_id);
}

template<typename T>
void UBO<T>::upload(const T& data) {
    glBindBuffer(GL_UNIFORM_BUFFER, _id);
    // Overwrite the existing buffer with the new struct data
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(T), &data);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

template<typename T>
void UBO<T>::bindTo(GLuint bindingIndex) const {
    glBindBufferBase(GL_UNIFORM_BUFFER, bindingIndex, _id);
}

template<typename T>
GLuint UBO<T>::getID() const {
    return _id;
}

// Explicit instantiation for FluidConfigUBO
#include "fluid/PBFluids.h"

template class UBO<FluidConfigUBO>;