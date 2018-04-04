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

  vec4 diffuse_colour = texture2D(s_diffuse_colour, iFS_TexCoord.xy);

	float alpha = check_alpha(diffuse_colour.a);

  diffuse_colour.rgb = _linear(diffuse_colour.rgb);

  gl_FragColor = vec4(_gamma(diffuse_colour.rgb), alpha);
}
