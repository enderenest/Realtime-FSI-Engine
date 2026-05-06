#ifndef COMPUTE_SHADER_H
#define COMPUTE_SHADER_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>

#include "core/Types.h"

class ComputeShader
{
public:
	ComputeShader(const char* computeFile);

	~ComputeShader();

	void use() const;

	void dispatch(U32 groupsX, U32 groupsY = 1, U32 groupsZ = 1) const;

	void setInt(const char* name, I32 value) const;

	void setFloat(const char* name, F32 value) const;

	void setUint(const char* name, const U32 value) const;

	void setVec3(const char* name, F32 x, F32 y, F32 z) const;

	void wait() const;

	U32 getID();

private:
	U32 _id;

	std::string loadShaderSource(const char* filePath) const;

	void checkCompileErrors(GLuint object, const std::string& type, const char* filename) const;
};

#endif