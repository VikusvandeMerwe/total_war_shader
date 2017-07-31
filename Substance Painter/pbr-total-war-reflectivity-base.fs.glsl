// Shader entry point
vec4 shade(V2F inputs)
{
  vec4 reflectivity_p = texture(s_reflectivity, inputs.tex_coord.xy);

  reflectivity_p.rgb = _linear(reflectivity_p.rgb);

  return vec4(reflectivity_p.rrr, 1.0);
}
