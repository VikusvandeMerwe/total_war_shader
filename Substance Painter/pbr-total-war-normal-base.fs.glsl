// Shader entry point
vec4 shade(V2F inputs)
{
  vec3 N = normalSwizzle(texture(s_normal_map, inputs.tex_coord.xy).rgb);

  return vec4(N.rgb, 1.0);
}
