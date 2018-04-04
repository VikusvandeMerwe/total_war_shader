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

  vec3 pI = normalize(vMatrixI[3].xyz - iFS_Wpos);
	mat3 basis = MAXTBN;
  vec3 Np = (texture2D(s_normal_map, iFS_TexCoord.xy)).rgb;
  Np.g = 1.0 - Np.g;
	vec3 N = normalSwizzle((Np.rgb * 2.0) - 1.0);

  if (true)
  {
    vec3 N2p = (texture2D(s_d_normal, iFS_TexCoord.xy * 1.0)).rgb;
    N2p.g = 1.0 - N2p.g;
    vec3 N2 = normalSwizzle((N2p.rgb * 2.0) - 1.0);
    N = vec3(N.x + (N2.x * 1.0), N.y + (N2.y * 1.0), N.z);
  }

	vec3 nN = normalize(basis * N);
	vec3 refl = reflect(pI, nN);
	refl.z = -refl.z;
	vec3 env = get_environment_colour_UPDATED(refl, 0.0);

  gl_FragColor = vec4(_gamma(env.rgb), 1.0);
}
