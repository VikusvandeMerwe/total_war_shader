//////////////////////////////// Fragment shader

#include "computation.fs.glsl"

/////////////////////////
// Fragment Shader
/////////////////////////

void main()
{
  vec4 specular_p = texture2D(s_specular_colour, iFS_TexCoord.xy );

  specular_p.rgb = _linear(specular_p.rgb);

  gl_FragColor = vec4(_gamma(specular_p.rgb), 1.0);
}
