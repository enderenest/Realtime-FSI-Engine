#include "opengl/SSBO.h"

template<typename T>
SSBO<T>::SSBO(size_t count, GLenum usage)
    : _count(count)
{
    glGenBuffers(1, &_id);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _id);
    glBufferData(
        GL_SHADER_STORAGE_BUFFER,
        _count * sizeof(T),
        nullptr,         // no initial data
        usage
    );
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

template<typename T>
SSBO<T>::~SSBO() {
    glDeleteBuffers(1, &_id);
}

template<typename T>
void SSBO<T>::upload(const std::vector<T>& data) {
    if (data.size() != _count) {
        _count = data.size();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _id);
        glBufferData(GL_SHADER_STORAGE_BUFFER, _count * sizeof(T), data.data(), GL_DYNAMIC_DRAW);
    }
    else {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _id);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, _count * sizeof(T), data.data());
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

template<typename T>
void SSBO<T>::bindTo(GLuint bindingIndex) const {
    glBindBufferBase(
        GL_SHADER_STORAGE_BUFFER,
        bindingIndex,
        _id
    );
}

template<typename T>
T* SSBO<T>::map(GLbitfield access) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _id);
    return static_cast<T*>(
        glMapBuffer(GL_SHADER_STORAGE_BUFFER, access)
        );
}

template<typename T>
void SSBO<T>::unmap() {
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
}

template<typename T>
size_t SSBO<T>::count() const {
    return _count;
}

template<typename T>
GLuint SSBO<T>::getID() const {
    return _id;
}


// Explicit instantiations for our types
#include "fluid/Particle.h"
#include "core/Types.h" 

template class SSBO<Particle>;
template class SSBO<PVec4>;
template class SSBO<UVec2>;
template class SSBO<IVec2>;
template class SSBO<U32>;