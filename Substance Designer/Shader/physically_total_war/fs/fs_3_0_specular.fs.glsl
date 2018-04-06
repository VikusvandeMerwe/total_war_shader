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

  vec4 specular_p = texture2D(s_specular_color, iFS_UV.xy );

  specular_p.rgb = _linear(specular_p.rgb);

  gl_FragColor = vec4(specular_p.rgb, 1.0);
}
