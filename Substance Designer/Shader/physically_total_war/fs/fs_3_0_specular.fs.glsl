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

  vec4 specular_p = texture2D(s_specular_colour, iFS_TexCoord.xy );

  specular_p.rgb = _linear(specular_p.rgb);

  gl_FragColor = vec4(_gamma(specular_p.rgb), 1.0);
}
