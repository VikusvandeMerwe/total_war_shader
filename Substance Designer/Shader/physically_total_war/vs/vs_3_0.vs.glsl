/////////////////////////////// Vertex Shader
#version 120

attribute vec4 iVS_Position;
attribute vec4 iVS_Normal;
attribute vec2 iVS_TexCoord;
attribute vec4 iVS_Tangent;
attribute vec4 iVS_Bitangent;
attribute vec4 iVS_Color;

varying vec3 iFS_Nml;
varying vec2 iFS_TexCoord;
varying vec3 iFS_Tgt;
varying vec3 iFS_Btgt;
varying vec3 iFS_Wpos;
varying vec4 iFS_Color;

uniform mat4 wMatrix;
uniform mat4 wvpMatrix;
uniform mat4 witMatrix;

void main()
{
	gl_Position = wvpMatrix * iVS_Position;

	// iFS_TexCoord.y += 1.0;
	// iFS_I = normalize(vMatrix[3] - (wMatrix * iVS_Position)).xyz;
	// iFS_Tgt = mat3(wMatrix) * iVS_Tangent;
	// iFS_Btgt = mat3(wMatrix) * iVS_Bitangent;
	// iFS_Nml = mat3(wMatrix) * iVS_Normal;
	iFS_Nml = normalize((witMatrix * iVS_Normal).xyz);
	iFS_TexCoord = iVS_TexCoord;
	iFS_Tgt = normalize((witMatrix * iVS_Tangent).xyz);
	iFS_Btgt = normalize((witMatrix * iVS_Bitangent).xyz);
	iFS_Wpos = (wMatrix * iVS_Position).xyz;
	iFS_Color = iVS_Color;
}
