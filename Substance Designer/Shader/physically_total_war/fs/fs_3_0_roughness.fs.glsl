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

  vec4 roughness_p = texture2D(s_smoothness, iFS_TexCoord.xy);

  roughness_p.rgb = _linear(roughness_p.rgb);

  gl_FragColor = vec4(_gamma(roughness_p.rrr), 1.0);
}
