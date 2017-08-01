//////////////////////////////// Fragment shader

#include "computation.fs.glsl"

/////////////////////////
// Fragment Shader
/////////////////////////

void main()
{
  vec3 ao = texture2D(s_ambient_occlusion, iFS_TexCoord.zw).rgb;

  gl_FragColor = vec4(ao.rgb, 1.0);
}
