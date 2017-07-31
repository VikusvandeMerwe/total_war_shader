// Shader entry point
vec4 shade(V2F inputs)
{
  vec3 ao = texture(s_ambient_occlusion, inputs.tex_coord.xy).rgb;

  return vec4(ao.rgb, 1.0);
}
