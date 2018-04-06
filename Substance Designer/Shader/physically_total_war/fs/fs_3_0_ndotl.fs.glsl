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

  vec4 diffuse_color = texture2D(s_diffuse_color, iFS_UV.xy);
  vec3 light_vector = normalize(light_position0.xyz -  iFS_PointWS);

  float alpha = check_alpha(diffuse_color.a);

  mat3 basis = MAXTBN;
  vec4 Np = texture2D(s_normal_map, iFS_UV.xy);
  Np.g = 1.0 - Np.g;
  vec3 N = normalSwizzle_UPDATED((Np.rgb * 2.0) - 1.0);
  vec3 pixel_normal = normalize(basis * normalize(N));

  vec3 ndotl = vec3(clamp(dot(pixel_normal, light_vector), 0.0, 1.0));

  gl_FragColor = vec4(ndotl, alpha);
}
