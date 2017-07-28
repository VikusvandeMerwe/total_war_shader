//////////////////////////////// Fragment shader

#include "computation.fs.glsl"

/////////////////////////
// Fragment Shader
/////////////////////////

void main()
{
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

	vec3 nN = ((normalize(basis * N)) * 0.5) + 0.5;

  gl_FragColor = vec4(nN.rgb, 1.0);
}
