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

  vec4 reflectivity_p = texture2D(s_reflectivity, iFS_UV.xy);

  reflectivity_p.rgb = _linear(reflectivity_p.rgb);

  gl_FragColor = vec4(reflectivity_p.rgb, 1.0);
}
