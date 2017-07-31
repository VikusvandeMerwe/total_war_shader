// Shader entry point
vec4 shade(V2F inputs)
{
  vec4 Ct = texture(s_diffuse_colour, inputs.tex_coord.xy);

  return vec4(inputs.color[0].rgb, Ct.a);
}
