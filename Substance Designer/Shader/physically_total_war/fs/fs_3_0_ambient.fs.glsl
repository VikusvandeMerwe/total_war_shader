//////////////////////////////// Fragment shader
#version 120
#extension GL_ARB_shader_texture_lod : require

#include "computation.fs.glsl"

void main()
{
  if (f_version != internal_version)
  {
    discard;
  }

  mat3 basis = MAXTBN;
  vec3 Np = (texture2D(s_normal_map, iFS_UV.xy)).rgb;
  Np.g = 1.0 - Np.g;
	vec3 N = normalSwizzle_UPDATED((Np.rgb * 2.0) - 1.0);

	vec3 nN = normalize(basis * N);

  gl_FragColor = vec4(cube_ambient(nN).rgb, 1.0);
}
