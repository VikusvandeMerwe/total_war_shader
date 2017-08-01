//////////////////////////////// Fragment shader

#include "computation.fs.glsl"

/////////////////////////
// Fragment Shader
/////////////////////////

void main()
{
  vec4 diffuse_colour = texture2D(s_diffuse_colour, iFS_TexCoord.xy );

	alpha_test(diffuse_colour.a);

  diffuse_colour.rgb = _linear(diffuse_colour.rgb);

  gl_FragColor = vec4(_gamma(diffuse_colour.rgb), 1.0);
}
