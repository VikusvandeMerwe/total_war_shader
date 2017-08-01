//////////////////////////////// Fragment shader

#include "computation.fs.glsl"

/////////////////////////
// Fragment Shader
/////////////////////////

void main()
{
  vec4 reflectivity_p = texture2D(s_reflectivity, iFS_TexCoord.xy);

  reflectivity_p.rgb = _linear(reflectivity_p.rgb);

  gl_FragColor = vec4(_gamma(reflectivity_p.rrr), 1.0);
}
