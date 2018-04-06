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

  vec3 pI = -normalize(viewInverseMatrix[3].xyz - iFS_PointWS);

	mat3 basis = MAXTBN;
  vec3 Np = (texture2D(s_normal_map, iFS_UV.xy)).rgb;
  Np.g = 1.0 - Np.g;
	vec3 N = normalSwizzle_UPDATED((Np.rgb * 2.0) - 1.0);

	vec3 nN = normalize(basis * normalize(N));
	vec3 refl = reflect(pI, nN);
	vec3 env = get_environment_color_UPDATED(rotate(refl, f_environment_rotation), 0.0);

  gl_FragColor = vec4(env.rgb, 1.0);
}
