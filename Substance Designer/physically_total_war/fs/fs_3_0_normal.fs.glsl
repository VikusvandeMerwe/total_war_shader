//////////////////////////////// Fragment shader

#include "computation.fs.glsl"

/////////////////////////
// Fragment Shader
/////////////////////////

void main()
{
  vec3 N = normalSwizzle(texture2D(s_normal_map, iFS_TexCoord.xy).rgb);

  gl_FragColor = vec4(N.rgb, 1.0);
}
