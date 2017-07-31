// Shader entry point
vec4 shade(V2F inputs)
{
  vec4 faction_p = texture(s_mask1, inputs.tex_coord.xy);

  return vec4(faction_p.rrr, 1.0);
}
