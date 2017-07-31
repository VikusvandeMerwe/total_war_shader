// Shader entry point
vec4 shade(V2F inputs)
{
  vec4 roughness_p = texture(s_smoothness, inputs.tex_coord.xy);

  roughness_p.rgb = _linear(roughness_p.rgb);

  return vec4(roughness_p.rrr, 1.0);
}
