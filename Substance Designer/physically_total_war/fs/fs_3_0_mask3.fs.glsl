//////////////////////////////// Fragment shader

#include "computation.fs.glsl"

/////////////////////////
// Fragment Shader
/////////////////////////

void main()
{
  vec4 faction_p = texture2D(s_mask3, iFS_TexCoord.xy);

  gl_FragColor = vec4(faction_p.rrr, 1.0);
}
