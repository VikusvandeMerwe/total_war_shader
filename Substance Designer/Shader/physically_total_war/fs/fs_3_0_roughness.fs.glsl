//////////////////////////////// Fragment shader

#include "computation.fs.glsl"

/////////////////////////
// Fragment Shader
/////////////////////////

void main()
{
  vec4 roughness_p = texture2D(s_smoothness, iFS_TexCoord.xy);

  roughness_p.rgb = _linear(roughness_p.rgb);

  gl_FragColor = vec4(_gamma(roughness_p.rrr), 1.0);
}
