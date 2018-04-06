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

  vec4 faction_p = texture2D(s_mask3, iFS_UV.xy);

  faction_p.rgb = _linear(faction_p.rgb);

  gl_FragColor = vec4(faction_p.rgb, 1.0);
}
