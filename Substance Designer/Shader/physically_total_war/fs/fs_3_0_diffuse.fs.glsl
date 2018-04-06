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

  vec4 diffuse_color = texture2D(s_diffuse_color, iFS_UV.xy);

	float alpha = check_alpha(diffuse_color.a);

  diffuse_color.rgb = _linear(diffuse_color.rgb);

  gl_FragColor = vec4(diffuse_color.rgb, alpha);
}
