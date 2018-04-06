/////////////////////////////// Vertex Shader
#version 120

attribute vec4 iVS_Position;
attribute vec4 iVS_Normal;
attribute vec2 iVS_UV;
attribute vec4 iVS_Tangent;
attribute vec4 iVS_Bitangent;
attribute vec4 iVS_VertexColor;

varying vec3 iFS_Normal;
varying vec2 iFS_UV;
varying vec3 iFS_Tangent;
varying vec3 iFS_Bitangent;
varying vec3 iFS_PointWS;
varying vec4 iFS_VertexColor;

uniform mat4 worldMatrix;
uniform mat4 worldViewProjMatrix;
uniform mat4 worldInverseTransposeMatrix;

void main()
{
	gl_Position = worldViewProjMatrix * iVS_Position;

	iFS_Normal = normalize((worldInverseTransposeMatrix * iVS_Normal).xyz);
	iFS_UV = iVS_UV;
	iFS_Tangent = normalize((worldInverseTransposeMatrix * iVS_Tangent).xyz);
	iFS_Bitangent = normalize((worldInverseTransposeMatrix * iVS_Bitangent).xyz);
	iFS_PointWS = (worldMatrix * iVS_Position).xyz;
	iFS_VertexColor = iVS_VertexColor;
}
