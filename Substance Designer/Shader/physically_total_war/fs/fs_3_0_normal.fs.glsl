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

  vec3 N = normalSwizzle(texture2D(s_normal_map, iFS_TexCoord.xy).rgb);

  gl_FragColor = vec4(N.rgb, 1.0);
}
