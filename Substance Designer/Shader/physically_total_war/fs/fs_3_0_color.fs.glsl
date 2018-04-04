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

  vec4 Ct = texture2D(s_diffuse_colour, iFS_TexCoord.xy);

  gl_FragColor = vec4(iFS_Color.rgb, Ct.a);
}
