#include <iostream>

#include "RndrSample.h"
#include "RndrGlut.h"
#include "RndrModeling.h"
#include "RndrProjection.h"
#include "RndrAliasing.h"
#include "RndrLighting.h"
#include "RndrTexture.h"
#include "CRndrGlfw.h"

enum GL_RNDR_TYPE
{
	// GLUT
	GL_RNDR_SAMPLE = 0,
	GL_RNDR_GLUT,
	GL_RNDR_MODELING,
	GL_RNDR_PROJECTION,
	GL_RNDR_ALIASING,
	GL_RNDR_LIGHTING,
	GL_RNDR_TEXTURE,

	// GLFW
	GL_RNDR_GLFW,

	// Num
	GL_RNDR_NUM
};

int main(int argc, char **argv)
{
	GL_RNDR_TYPE eType = GL_RNDR_GLFW;
	switch (eType)
	{
	case GL_RNDR_SAMPLE: { CRndrSample rndr; rndr.Run(); } break;
	case GL_RNDR_GLUT: { CRndrGlut rndr; rndr.Run(); } break;
	case GL_RNDR_MODELING: { CRndrModeling rndr; rndr.Run(); } break;
	case GL_RNDR_PROJECTION: { CRndrProjection rndr; rndr.Run(); } break;
	case GL_RNDR_ALIASING: { CRndrAliasing rndr; rndr.Run(); } break;
	case GL_RNDR_LIGHTING: { CRndrLighting rndr; rndr.Run(); } break;
	case GL_RNDR_TEXTURE: { CRndrTexture rndr; rndr.Run(); } break;
	case GL_RNDR_GLFW: { CRndrGlfw rndr; rndr.Run(); } break;
	}

	return 0;
}