//////////////////////////////// Fragment shader

#include "computation.fs.glsl"

/////////////////////////
// Fragment Shader
/////////////////////////

void main()
{
  vec4 Ct = texture2D(s_diffuse_colour, iFS_TexCoord.xy);

  gl_FragColor = vec4(iFS_Color.rgb, Ct.a);
}
