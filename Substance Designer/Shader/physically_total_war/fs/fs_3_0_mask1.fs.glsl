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

  vec4 faction_p = texture2D(s_mask1, iFS_TexCoord.xy);

  gl_FragColor = vec4(faction_p.rrr, 1.0);
}
