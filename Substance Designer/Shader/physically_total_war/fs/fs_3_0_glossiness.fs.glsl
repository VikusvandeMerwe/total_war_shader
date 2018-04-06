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

  vec4 glossiness_p = texture2D(s_smoothness, iFS_UV.xy);

  glossiness_p.rgb = _linear(glossiness_p.rgb);

  gl_FragColor = vec4(glossiness_p.rgb, 1.0);
}
