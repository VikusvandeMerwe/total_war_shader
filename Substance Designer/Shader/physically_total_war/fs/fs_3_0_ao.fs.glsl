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

  vec3 ao = texture2D(s_ambient_occlusion, iFS_UV.xy).rgb;

  ao.rgb = _linear(ao.rgb);

  gl_FragColor = vec4(ao.rgb, 1.0);
}
