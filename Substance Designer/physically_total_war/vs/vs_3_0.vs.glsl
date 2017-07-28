/////////////////////////////// Vertex Shader
#version 120

attribute vec4 iVS_Position;
attribute vec3 iVS_Normal;
attribute vec3 iVS_Tangent;
attribute vec3 iVS_Bitangent;
attribute vec4 iVS_TexCoord0;
attribute vec4 iVS_TexCoord1;
attribute vec3 iVS_Color;
attribute float iVS_Alpha;

varying vec4 iFS_TexCoord;
varying vec3 iFS_I;
varying vec3 iFS_Tgt;
varying vec3 iFS_Btgt;
varying vec3 iFS_Nml;
varying vec3 iFS_Wpos;
varying vec4 iFS_Color;

uniform mat4 wvpMatrix;
uniform mat4 wMatrix;
uniform mat4 vMatrix;

void main()
{
	gl_Position = wvpMatrix * iVS_Position;
	iFS_TexCoord.xy = iVS_TexCoord0.xy;
	iFS_TexCoord.zw = iVS_TexCoord1.xy;

	iFS_TexCoord.y += 1.0;
	iFS_TexCoord.w += 1.0;

	iFS_I = normalize(vMatrix[3] - (wMatrix * iVS_Position)).xyz;
	iFS_Tgt = (wMatrix * vec4(iVS_Tangent.xyz, 0.0)).xyz;
	iFS_Btgt = (wMatrix * vec4(iVS_Bitangent.xyz, 0.0)).xyz;
	iFS_Nml = (wMatrix * vec4(iVS_Normal.xyz, 0.0)).xyz;
	iFS_Wpos = (wMatrix * iVS_Position).xyz;
	iFS_Color.rgb = iVS_Color.rgb;
	iFS_Color.a = iVS_Alpha;
}
